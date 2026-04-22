package backend

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/robcowart/aiproxy/pkg/metrics"
	"github.com/robcowart/aiproxy/pkg/schema"
	"go.uber.org/zap"
)

// Forwarder issues translated backend HTTP requests, injecting the instance's Authorization header, and records logs
// and metrics per call.
type Forwarder struct {
	log     *zap.Logger
	metrics *metrics.Metrics
}

// NewForwarder builds a Forwarder. log and m may be nil (logging/metrics then become no-ops).
func NewForwarder(log *zap.Logger, m *metrics.Metrics) *Forwarder {
	if log == nil {
		log = zap.NewNop()
	}
	return &Forwarder{log: log, metrics: m}
}

// Do issues breq to inst and returns the response along with a finish closure. The caller is responsible for closing
// resp.Body and for invoking finish(usage) exactly once — typically via defer — to emit the backend-request access-log
// line. Pass the parsed *schema.Usage when available (nil otherwise); when non-nil, prompt_tokens and completion_tokens
// fields are added to the log. Metrics (BackendRequests, BackendRequestDuration) are recorded immediately on
// response-header return, so finish only governs log emission. On error the returned response and finish are nil and a
// warn-level log has already been emitted.
func (f *Forwarder) Do(ctx context.Context, pool *Pool, inst *Instance, breq *schema.BackendRequest) (*http.Response, func(*schema.Usage), error) {
	url := strings.TrimRight(inst.URL, "/") + breq.Path
	var body io.Reader
	if len(breq.Body) > 0 {
		body = bytes.NewReader(breq.Body)
	}
	req, err := http.NewRequestWithContext(ctx, breq.Method, url, body)
	if err != nil {
		return nil, nil, fmt.Errorf("new backend request: %w", err)
	}
	if breq.Headers != nil {
		for k, vs := range breq.Headers {
			for _, v := range vs {
				req.Header.Add(k, v)
			}
		}
	}
	if req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	if inst.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+inst.APIKey)
	}

	start := time.Now()
	if f.metrics != nil {
		f.metrics.BackendInflight.WithLabelValues(pool.Model, inst.URL).Inc()
		defer f.metrics.BackendInflight.WithLabelValues(pool.Model, inst.URL).Dec()
	}

	resp, err := inst.Client().Do(req)
	dur := time.Since(start)
	durMs := float64(dur.Nanoseconds()) / float64(time.Millisecond)
	if err != nil {
		f.log.Warn("backend request failed",
			zap.String("pool", pool.Model),
			zap.String("instance", inst.URL),
			zap.String("path", breq.Path),
			zap.Duration("duration", dur),
			zap.Error(err))
		if f.metrics != nil {
			f.metrics.BackendRequests.WithLabelValues(pool.Model, inst.URL, "error").Inc()
			f.metrics.BackendRequestDuration.WithLabelValues(pool.Model, inst.URL, "error").Add(durMs)
		}
		return nil, nil, err
	}
	if f.metrics != nil {
		statusLabel := HTTPStatusLabel(resp.StatusCode)
		f.metrics.BackendRequests.WithLabelValues(pool.Model, inst.URL, statusLabel).Inc()
		f.metrics.BackendRequestDuration.WithLabelValues(pool.Model, inst.URL, statusLabel).Add(durMs)
	}

	var finished bool
	finish := func(u *schema.Usage) {
		if finished {
			return
		}
		finished = true
		fields := []zap.Field{
			zap.String("pool", pool.Model),
			zap.String("instance", inst.URL),
			zap.String("method", breq.Method),
			zap.String("path", breq.Path),
			zap.Int("status", resp.StatusCode),
			zap.Duration("duration", dur),
		}
		if u != nil {
			fields = append(fields,
				zap.Int("prompt_tokens", u.PromptTokens),
				zap.Int("completion_tokens", u.CompletionTokens),
			)
		}
		f.log.Info("backend request", fields...)
	}
	return resp, finish, nil
}

// HTTPStatusLabel buckets an HTTP status code into a Prometheus-friendly class label ("2xx", "3xx", "4xx", "5xx", or
// "other" for codes outside the 200..599 range).
func HTTPStatusLabel(status int) string {
	switch {
	case status >= 200 && status < 300:
		return "2xx"
	case status >= 300 && status < 400:
		return "3xx"
	case status >= 400 && status < 500:
		return "4xx"
	case status >= 500:
		return "5xx"
	default:
		return "other"
	}
}
