package metrics

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew_ExposesAllCollectors(t *testing.T) {
	m := New()
	assert.NotNil(t, m.Registry)

	m.ClientRequests.WithLabelValues("p", "e", "200").Inc()
	m.ClientPromptTokens.WithLabelValues("p", "e").Add(123)
	m.ClientCompletionTokens.WithLabelValues("p", "e").Add(45)
	m.BackendPromptTokens.WithLabelValues("p", "i").Add(123)
	m.BackendCompletionTokens.WithLabelValues("p", "i").Add(45)
	m.BackendInflight.WithLabelValues("p", "i").Set(3)
	m.BackendUp.WithLabelValues("p", "i").Set(1)
	m.HealthChecks.WithLabelValues("p", "i", "ok").Inc()
	m.SessionsActive.WithLabelValues("p").Set(7)
	m.StreamActive.WithLabelValues("p").Set(2)

	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL)
	assert.NoError(t, err)
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	assert.NoError(t, err)
	out := string(b)

	for _, name := range []string{
		"aiproxy_client_requests_total",
		"aiproxy_client_prompt_tokens_total",
		"aiproxy_client_completion_tokens_total",
		"aiproxy_backend_prompt_tokens_total",
		"aiproxy_backend_completion_tokens_total",
		"aiproxy_backend_inflight",
		"aiproxy_backend_up",
		"aiproxy_health_checks_total",
		"aiproxy_sessions_active",
		"aiproxy_stream_active",
	} {
		assert.True(t, strings.Contains(out, name), "metric %q missing from /metrics output", name)
	}
}
