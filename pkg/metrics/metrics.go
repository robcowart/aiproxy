// Package metrics exposes Prometheus counters and gauges used by aiproxy, plus a /metrics HTTP handler.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds the full set of Prometheus collectors.
type Metrics struct {
	Registry *prometheus.Registry

	ClientRequests         *prometheus.CounterVec
	ClientRequestDuration  *prometheus.CounterVec
	ClientPromptTokens     *prometheus.CounterVec
	ClientCompletionTokens *prometheus.CounterVec

	BackendRequests         *prometheus.CounterVec
	BackendRequestDuration  *prometheus.CounterVec
	BackendPromptTokens     *prometheus.CounterVec
	BackendCompletionTokens *prometheus.CounterVec
	BackendInflight         *prometheus.GaugeVec
	BackendUp               *prometheus.GaugeVec
	HealthChecks            *prometheus.CounterVec

	SessionsActive *prometheus.GaugeVec
	StreamActive   *prometheus.GaugeVec
}

// New constructs a fresh metrics registry with all aiproxy collectors registered.
func New() *Metrics {
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector())
	reg.MustRegister(collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))

	m := &Metrics{
		Registry: reg,

		ClientRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_client_requests_total",
			Help: "Client-facing HTTP requests served, labeled by pool, endpoint, and HTTP status.",
		}, []string{"pool", "endpoint", "status"}),

		ClientRequestDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_client_request_duration_ms_total",
			Help: "Cumulative client-facing request duration in milliseconds, labeled by pool, endpoint, and HTTP status. Divide by aiproxy_client_requests_total (matching labels) to get average request duration.",
		}, []string{"pool", "endpoint", "status"}),

		ClientPromptTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_client_prompt_tokens_total",
			Help: "Cumulative prompt tokens reported by the backend for client-facing requests, labeled by pool and endpoint.",
		}, []string{"pool", "endpoint"}),

		ClientCompletionTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_client_completion_tokens_total",
			Help: "Cumulative completion tokens reported by the backend for client-facing requests, labeled by pool and endpoint.",
		}, []string{"pool", "endpoint"}),

		BackendRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_backend_requests_total",
			Help: "Requests forwarded to backend instances, labeled by pool, instance, and HTTP status.",
		}, []string{"pool", "instance", "status"}),

		BackendRequestDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_backend_request_duration_ms_total",
			Help: "Cumulative backend request duration in milliseconds, labeled by pool, instance, and HTTP status. Divide by aiproxy_backend_requests_total (matching labels) to get average backend request duration.",
		}, []string{"pool", "instance", "status"}),

		BackendPromptTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_backend_prompt_tokens_total",
			Help: "Cumulative prompt tokens attributed to backend instances, labeled by pool and instance.",
		}, []string{"pool", "instance"}),

		BackendCompletionTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_backend_completion_tokens_total",
			Help: "Cumulative completion tokens attributed to backend instances, labeled by pool and instance.",
		}, []string{"pool", "instance"}),

		BackendInflight: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "aiproxy_backend_inflight",
			Help: "Current in-flight requests per backend instance.",
		}, []string{"pool", "instance"}),

		BackendUp: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "aiproxy_backend_up",
			Help: "1 if the backend instance is healthy, 0 otherwise.",
		}, []string{"pool", "instance"}),

		HealthChecks: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "aiproxy_health_checks_total",
			Help: "Backend health-check attempts, labeled by pool, instance, and result.",
		}, []string{"pool", "instance", "result"}),

		SessionsActive: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "aiproxy_sessions_active",
			Help: "Active sticky sessions per pool.",
		}, []string{"pool"}),

		StreamActive: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "aiproxy_stream_active",
			Help: "Active streaming responses per pool.",
		}, []string{"pool"}),
	}

	reg.MustRegister(
		m.ClientRequests,
		m.ClientRequestDuration,
		m.ClientPromptTokens,
		m.ClientCompletionTokens,
		m.BackendRequests,
		m.BackendRequestDuration,
		m.BackendPromptTokens,
		m.BackendCompletionTokens,
		m.BackendInflight,
		m.BackendUp,
		m.HealthChecks,
		m.SessionsActive,
		m.StreamActive,
	)
	return m
}

// Handler returns an http.Handler exposing the Prometheus /metrics endpoint.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.Registry, promhttp.HandlerOpts{Registry: m.Registry})
}
