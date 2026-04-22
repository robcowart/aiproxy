package frontend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/robcowart/aiproxy/pkg/backend"
	"github.com/robcowart/aiproxy/pkg/config"
	"github.com/robcowart/aiproxy/pkg/schema"
)

func (f *Frontend) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req schema.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf("parse body: %v", err))
		return
	}
	pool := f.resolvePool(w, req.Model, config.EndpointChatCompletions)
	if pool == nil {
		return
	}

	sessionKey := sessionKey(r, pool.Model)
	inst, err := pool.Pick(sessionKey)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "no_backend", err.Error())
		return
	}
	defer pool.Release(inst)

	clientOptedOut := false
	if req.Stream {
		normalized, optedOut, nerr := schema.NormalizeStreamOptions(req.StreamOptions)
		if nerr != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", nerr.Error())
			return
		}
		req.StreamOptions = normalized
		clientOptedOut = optedOut
	}

	breq, err := pool.Translator.ChatBackendRequest(&req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "translate_request", err.Error())
		return
	}

	ctx := r.Context()
	if !req.Stream {
		start := time.Now()
		resp, finish, err := f.fwd.Do(ctx, pool, inst, breq)
		if err != nil {
			writeError(w, http.StatusBadGateway, "backend_error", err.Error())
			return
		}
		defer resp.Body.Close()
		var loggedUsage *schema.Usage
		defer func() { finish(loggedUsage) }()
		f.observeClient(pool, "chat_completions", resp.StatusCode, start)
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			writeError(w, http.StatusBadGateway, "backend_read", err.Error())
			return
		}
		if resp.StatusCode >= 400 {
			proxyError(w, resp.StatusCode, body)
			return
		}
		parsed, err := pool.Translator.ChatResponseFromBytes(body)
		if err != nil {
			writeError(w, http.StatusBadGateway, "translate_response", err.Error())
			return
		}
		loggedUsage = parsed.Usage
		f.observeTokens(pool, inst, "chat_completions", parsed.Usage)
		recordUsage(w, parsed.Usage)
		writeJSON(w, http.StatusOK, parsed)
		return
	}

	if f.metrics != nil {
		f.metrics.StreamActive.WithLabelValues(pool.Model).Inc()
		defer f.metrics.StreamActive.WithLabelValues(pool.Model).Dec()
	}

	resp, finish, err := f.fwd.Do(ctx, pool, inst, breq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "backend_error", err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		proxyError(w, resp.StatusCode, body)
		finish(nil)
		return
	}

	f.streamChat(ctx, w, pool, inst, resp.Body, clientOptedOut, finish)
}

func (f *Frontend) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req schema.EmbeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf("parse body: %v", err))
		return
	}
	pool := f.resolvePool(w, req.Model, config.EndpointEmbeddings)
	if pool == nil {
		return
	}

	inst, err := pool.Pick(sessionKey(r, pool.Model))
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "no_backend", err.Error())
		return
	}
	defer pool.Release(inst)

	breq, err := pool.Translator.EmbeddingsBackendRequest(&req)
	if writeTranslateErr(w, err) {
		return
	}
	body, finish := f.doAndRead(r.Context(), pool, inst, breq, w, "embeddings")
	if body == nil {
		return
	}
	var loggedUsage *schema.Usage
	defer func() { finish(loggedUsage) }()
	parsed, err := pool.Translator.EmbeddingsResponseFromBytes(body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "translate_response", err.Error())
		return
	}
	loggedUsage = parsed.Usage
	f.observeTokens(pool, inst, "embeddings", parsed.Usage)
	recordUsage(w, parsed.Usage)
	writeJSON(w, http.StatusOK, parsed)
}

func (f *Frontend) handleRerank(w http.ResponseWriter, r *http.Request) {
	var req schema.RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf("parse body: %v", err))
		return
	}
	pool := f.resolvePool(w, req.Model, config.EndpointRerank)
	if pool == nil {
		return
	}

	inst, err := pool.Pick(sessionKey(r, pool.Model))
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "no_backend", err.Error())
		return
	}
	defer pool.Release(inst)

	breq, err := pool.Translator.RerankBackendRequest(&req)
	if writeTranslateErr(w, err) {
		return
	}
	body, finish := f.doAndRead(r.Context(), pool, inst, breq, w, "rerank")
	if body == nil {
		return
	}
	var loggedUsage *schema.Usage
	defer func() { finish(loggedUsage) }()
	parsed, err := pool.Translator.RerankResponseFromBytes(body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "translate_response", err.Error())
		return
	}
	loggedUsage = parsed.Usage
	f.observeTokens(pool, inst, "rerank", parsed.Usage)
	recordUsage(w, parsed.Usage)
	writeJSON(w, http.StatusOK, parsed)
}

// writeTranslateErr maps an error from a translator's *BackendRequest builder to an HTTP response: 501 for
// schema.ErrUnsupportedEndpoint, 500 for anything else. Returns true when err was non-nil (and a response has been
// written), so callers can `if writeTranslateErr(w, err) { return }`.
func writeTranslateErr(w http.ResponseWriter, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, schema.ErrUnsupportedEndpoint) {
		writeError(w, http.StatusNotImplemented, "unsupported_endpoint", err.Error())
		return true
	}
	writeError(w, http.StatusInternalServerError, "translate_request", err.Error())
	return true
}

// resolvePool looks up the backend pool for model and verifies it serves endpoint. On mismatch it writes the
// appropriate error response (404 for unknown model, 400 for wrong endpoint) and returns nil.
func (f *Frontend) resolvePool(w http.ResponseWriter, model string, endpoint config.EndpointType) *backend.Pool {
	pool, ok := f.pools.Get(model)
	if !ok {
		writeError(w, http.StatusNotFound, "model_not_found", "unknown model: "+model)
		return nil
	}
	if pool.Endpoint != endpoint {
		writeError(w, http.StatusBadRequest, "wrong_endpoint", fmt.Sprintf("model %q serves %s", model, pool.Endpoint))
		return nil
	}
	return pool
}

// doAndRead forwards a simple request-response backend call. On success it returns the response body and a finish
// closure the caller must invoke (typically via defer with the parsed *schema.Usage) to emit the backend access-log
// line. On any error path, it writes a response to w, emits finish(nil) internally, and returns (nil, nil).
func (f *Frontend) doAndRead(ctx context.Context, pool *backend.Pool, inst *backend.Instance, breq *schema.BackendRequest, w http.ResponseWriter, endpoint string) ([]byte, func(*schema.Usage)) {
	start := time.Now()
	resp, finish, err := f.fwd.Do(ctx, pool, inst, breq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "backend_error", err.Error())
		return nil, nil
	}
	defer resp.Body.Close()
	f.observeClient(pool, endpoint, resp.StatusCode, start)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "backend_read", err.Error())
		finish(nil)
		return nil, nil
	}
	if resp.StatusCode >= 400 {
		proxyError(w, resp.StatusCode, body)
		finish(nil)
		return nil, nil
	}
	return body, finish
}

func (f *Frontend) observeClient(pool *backend.Pool, endpoint string, status int, start time.Time) {
	if f.metrics == nil {
		return
	}
	statusLabel := backend.HTTPStatusLabel(status)
	f.metrics.ClientRequests.WithLabelValues(pool.Model, endpoint, statusLabel).Inc()
	ms := float64(time.Since(start).Nanoseconds()) / float64(time.Millisecond)
	f.metrics.ClientRequestDuration.WithLabelValues(pool.Model, endpoint, statusLabel).Add(ms)
}

// observeTokens increments client- and backend-scoped token counters from a single canonical Usage. No-op when either
// metrics or usage is nil, or when inst is nil (callers may pass nil for paths without an attributable instance).
func (f *Frontend) observeTokens(pool *backend.Pool, inst *backend.Instance, endpoint string, u *schema.Usage) {
	if f.metrics == nil || u == nil {
		return
	}
	f.metrics.ClientPromptTokens.WithLabelValues(pool.Model, endpoint).Add(float64(u.PromptTokens))
	f.metrics.ClientCompletionTokens.WithLabelValues(pool.Model, endpoint).Add(float64(u.CompletionTokens))
	if inst != nil {
		f.metrics.BackendPromptTokens.WithLabelValues(pool.Model, inst.URL).Add(float64(u.PromptTokens))
		f.metrics.BackendCompletionTokens.WithLabelValues(pool.Model, inst.URL).Add(float64(u.CompletionTokens))
	}
}

// recordUsage stashes usage on the respRecorder (installed by requestLogger) so the access-log middleware can include
// token fields. Safely no-ops when w is not the expected recorder or usage is nil.
func recordUsage(w http.ResponseWriter, u *schema.Usage) {
	if u == nil {
		return
	}
	if rr, ok := w.(*respRecorder); ok {
		rr.setUsage(u)
	}
}

// sessionKey derives the sticky-session key from the request.
func sessionKey(r *http.Request, poolName string) string {
	if v := r.Header.Get("X-Session-Id"); v != "" {
		return "hdr:" + v + "|" + poolName
	}
	host, port, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
		port = ""
	}
	return "conn:" + host + ":" + port + "|" + poolName
}

// streamChat pipes a backend SSE chat stream through the pool's translator and writes canonical OpenAI SSE frames to w.
// Every chunk is probed for Usage (last-wins across the stream); all three built-in translators emit cumulative
// final-chunk usage, so last-wins is correct. A canonical usage-only chunk (choices empty, usage populated) is
// forwarded to the client only when clientOptedOut is false; either way the observed Usage is recorded on the access
// log recorder and token counters after the stream closes. finish (from Forwarder.Do) is invoked when the stream
// terminates so the backend access-log line includes the final prompt_tokens/completion_tokens.
func (f *Frontend) streamChat(ctx context.Context, w http.ResponseWriter, pool *backend.Pool, inst *backend.Instance, body io.ReadCloser, clientOptedOut bool, finish func(*schema.Usage)) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "no_flusher", "streaming not supported")
		finish(nil)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	reader, err := pool.Translator.NewChatStreamReader(body)
	if err != nil {
		_ = writeSSEError(w, flusher, err)
		finish(nil)
		return
	}
	defer reader.Close()

	var lastUsage *schema.Usage
	defer func() {
		if lastUsage != nil {
			f.observeTokens(pool, inst, "chat_completions", lastUsage)
			recordUsage(w, lastUsage)
		}
		finish(lastUsage)
	}()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		chunks, done, err := reader.Next()
		if err != nil {
			_ = writeSSEError(w, flusher, err)
			return
		}
		for _, c := range chunks {
			var probe struct {
				Choices []json.RawMessage `json:"choices"`
				Usage   *schema.Usage     `json:"usage"`
			}
			if perr := json.Unmarshal(c, &probe); perr == nil && probe.Usage != nil {
				lastUsage = probe.Usage
			}
			if clientOptedOut && len(probe.Choices) == 0 && probe.Usage != nil {
				continue
			}
			_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		if done {
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
	}
}

func writeSSEError(w http.ResponseWriter, flusher http.Flusher, err error) error {
	payload := map[string]any{
		"error": map[string]string{
			"type":    "stream_error",
			"message": err.Error(),
		},
	}
	b, _ := json.Marshal(payload)
	_, werr := fmt.Fprintf(w, "data: %s\n\n", b)
	flusher.Flush()
	return werr
}

// writeJSON serializes v as a JSON response body.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError writes an OpenAI-style error object.
func writeError(w http.ResponseWriter, status int, typ, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{
			"type":    typ,
			"message": message,
		},
	})
}

// proxyError forwards a backend error body to the client with the original status code. If body is not JSON, wrap it in
// a standard error object.
func proxyError(w http.ResponseWriter, status int, body []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if json.Valid(body) {
		_, _ = w.Write(body)
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"type":    "backend_error",
			"message": string(body),
		},
	})
}
