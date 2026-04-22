// Package frontend implements the client-facing HTTP server: OpenAI-compatible routes, authentication, structured
// request logs, Prometheus metrics surfacing, and SSE streaming passthrough for chat completions.
package frontend

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/robcowart/aiproxy/pkg/backend"
	"github.com/robcowart/aiproxy/pkg/config"
	"github.com/robcowart/aiproxy/pkg/metrics"
	"github.com/robcowart/aiproxy/pkg/schema"
	"go.uber.org/zap"
)

// modelsCacheTTL bounds how long per-pool /v1/models metadata is cached before the frontend re-probes the backend.
const modelsCacheTTL = 60 * time.Second

// modelsProbeTimeout bounds a single backend /v1/models probe.
const modelsProbeTimeout = 5 * time.Second

type modelsCacheEntry struct {
	info   schema.ModelInfo
	expiry time.Time
}

// Frontend is the aiproxy HTTP server.
type Frontend struct {
	cfg     *config.Config
	pools   *backend.Registry
	fwd     *backend.Forwarder
	log     *zap.Logger
	metrics *metrics.Metrics
	handler http.Handler

	modelsMu    sync.Mutex
	modelsCache map[string]modelsCacheEntry
}

// NewFrontend wires routes, middleware, and dependencies.
func NewFrontend(cfg *config.Config, pools *backend.Registry, fwd *backend.Forwarder, log *zap.Logger, m *metrics.Metrics) *Frontend {
	if log == nil {
		log = zap.NewNop()
	}
	f := &Frontend{
		cfg:         cfg,
		pools:       pools,
		fwd:         fwd,
		log:         log,
		metrics:     m,
		modelsCache: make(map[string]modelsCacheEntry),
	}
	f.handler = f.buildHandler()
	return f
}

// Handler returns the fully composed http.Handler (auth + logging + routes).
func (f *Frontend) Handler() http.Handler { return f.handler }

// Addr returns the host:port string derived from config.
func (f *Frontend) Addr() string {
	return fmt.Sprintf("%s:%d", f.cfg.Server.Host, f.cfg.Server.Port)
}

// ShutdownTimeout bounds how long the server waits for in-flight requests to complete during graceful shutdown.
const ShutdownTimeout = 10 * time.Second

// ListenAndServe starts the HTTP(S) listener and blocks until ctx is cancelled or the listener fails. On ctx
// cancellation the server is shut down gracefully within ShutdownTimeout.
func (f *Frontend) ListenAndServe(ctx context.Context) error {
	httpSrv := &http.Server{
		Addr:              f.Addr(),
		Handler:           f.handler,
		ReadHeaderTimeout: 10 * time.Second,
	}
	if f.cfg.Server.TLS.Enabled {
		httpSrv.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}

	errCh := make(chan error, 1)
	go func() {
		f.log.Info("http listener starting",
			zap.String("addr", httpSrv.Addr),
			zap.Bool("tls", f.cfg.Server.TLS.Enabled))
		if f.cfg.Server.TLS.Enabled {
			errCh <- httpSrv.ListenAndServeTLS(f.cfg.Server.TLS.CertFile, f.cfg.Server.TLS.KeyFile)
		} else {
			errCh <- httpSrv.ListenAndServe()
		}
	}()

	select {
	case <-ctx.Done():
		f.log.Info("shutdown requested")
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("http listener: %w", err)
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), ShutdownTimeout)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		f.log.Warn("graceful shutdown error", zap.Error(err))
	}
	return nil
}

func (f *Frontend) buildHandler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("GET /health", http.HandlerFunc(f.handleHealth))
	if f.metrics != nil {
		mux.Handle("GET /metrics", f.metrics.Handler())
	}
	mux.Handle("GET /v1/models", http.HandlerFunc(f.handleListModels))
	mux.Handle("POST /v1/chat/completions", http.HandlerFunc(f.handleChatCompletions))
	mux.Handle("POST /v1/embeddings", http.HandlerFunc(f.handleEmbeddings))
	mux.Handle("POST /v1/rerank", http.HandlerFunc(f.handleRerank))

	skip := map[string]bool{"/health": true, "/metrics": true}
	var h http.Handler = mux
	h = requireAPIKey(f.cfg.Server.APIKey, skip, h)
	h = f.requestLogger(h)
	return h
}

// requestLogger wraps h with a structured zap access log.
func (f *Frontend) requestLogger(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &respRecorder{ResponseWriter: w, status: http.StatusOK}
		h.ServeHTTP(rw, r)
		fields := []zap.Field{
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.String("client", r.RemoteAddr),
			zap.Int("status", rw.status),
			zap.Int("bytes", rw.bytes),
			zap.Duration("duration", time.Since(start)),
		}
		if rw.usage != nil {
			fields = append(fields,
				zap.Int("prompt_tokens", rw.usage.PromptTokens),
				zap.Int("completion_tokens", rw.usage.CompletionTokens),
			)
		}
		f.log.Info("client request", fields...)
	})
}

type respRecorder struct {
	http.ResponseWriter
	status      int
	bytes       int
	wroteHeader bool
	usage       *schema.Usage
}

// setUsage records the token usage for this response so the access-log middleware can include it. nil is ignored so
// handlers can call it unconditionally.
func (r *respRecorder) setUsage(u *schema.Usage) {
	if u == nil {
		return
	}
	r.usage = u
}

func (r *respRecorder) WriteHeader(code int) {
	if !r.wroteHeader {
		r.status = code
		r.wroteHeader = true
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *respRecorder) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.status = http.StatusOK
		r.wroteHeader = true
	}
	n, err := r.ResponseWriter.Write(b)
	r.bytes += n
	return n, err
}

// Flush passes the Flush call through when the underlying ResponseWriter implements http.Flusher, which is required for
// SSE streaming.
func (r *respRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// handleHealth is a simple liveness probe for the proxy itself.
func (f *Frontend) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

// handleListModels aggregates model metadata across all pools into an OpenAI /v1/models response, enriching each entry
// with the `created` timestamp and any provider-specific `meta` block returned by the backend (e.g., llama.cpp).
// Metadata is cached per pool with modelsCacheTTL; on probe failure the response falls back to the minimal {id, object,
// owned_by} entry.
func (f *Frontend) handleListModels(w http.ResponseWriter, r *http.Request) {
	type resp struct {
		Object string             `json:"object"`
		Data   []schema.ModelInfo `json:"data"`
	}
	out := resp{Object: "list", Data: make([]schema.ModelInfo, 0, len(f.pools.All()))}
	for _, p := range f.pools.All() {
		out.Data = append(out.Data, f.resolveModelInfo(r.Context(), p))
	}
	sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].ID < out.Data[j].ID })
	writeJSON(w, http.StatusOK, out)
}

// resolveModelInfo returns cached metadata for a pool if fresh, otherwise probes the backend once and caches the result
// (success or fallback).
func (f *Frontend) resolveModelInfo(ctx context.Context, p *backend.Pool) schema.ModelInfo {
	now := time.Now()
	f.modelsMu.Lock()
	if ent, ok := f.modelsCache[p.Model]; ok && now.Before(ent.expiry) {
		f.modelsMu.Unlock()
		return ent.info
	}
	f.modelsMu.Unlock()

	info := f.fetchModelInfo(ctx, p)

	f.modelsMu.Lock()
	f.modelsCache[p.Model] = modelsCacheEntry{info: info, expiry: time.Now().Add(modelsCacheTTL)}
	f.modelsMu.Unlock()
	return info
}

// fetchModelInfo probes one healthy backend instance in the pool for its /v1/models listing and extracts the entry
// matching the pool's configured model. On any failure it returns the minimal fallback info.
func (f *Frontend) fetchModelInfo(ctx context.Context, p *backend.Pool) schema.ModelInfo {
	fallback := schema.ModelInfo{ID: p.Model, Object: "model", OwnedBy: string(p.Schema)}

	inst := p.FirstHealthy()
	if inst == nil {
		return fallback
	}
	breq, err := p.Translator.ModelsBackendRequest()
	if err != nil {
		return fallback
	}
	probeCtx, cancel := context.WithTimeout(ctx, modelsProbeTimeout)
	defer cancel()
	resp, finish, err := f.fwd.Do(probeCtx, p, inst, breq)
	if err != nil {
		return fallback
	}
	defer func() { _ = resp.Body.Close() }()
	defer finish(nil)
	body, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode >= 400 {
		return fallback
	}
	models, err := p.Translator.ModelsResponseFromBytes(body)
	if err != nil {
		f.log.Debug("models probe parse failed",
			zap.String("pool", p.Model), zap.String("instance", inst.URL), zap.Error(err))
		return fallback
	}
	for _, m := range models {
		if m.ID != p.Model {
			continue
		}
		if m.Object == "" {
			m.Object = "model"
		}
		if m.OwnedBy == "" {
			m.OwnedBy = string(p.Schema)
		}
		return m
	}
	return fallback
}
