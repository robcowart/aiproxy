package backend

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/robcowart/aiproxy/pkg/config"
	"github.com/robcowart/aiproxy/pkg/logging"
	"github.com/robcowart/aiproxy/pkg/metrics"
	"github.com/robcowart/aiproxy/pkg/schema"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"
)

func poolConfig(t *testing.T, model string, urls ...string) config.PoolConfig {
	t.Helper()
	pc := config.PoolConfig{
		Model:               model,
		Endpoint:            config.EndpointChatCompletions,
		Schema:              config.SchemaLlamaCPP,
		SessionTimeout:      60,
		HealthCheckInterval: 30,
	}
	for _, u := range urls {
		pc.Instances = append(pc.Instances, config.InstanceConfig{URL: u, APIKey: "k"})
	}
	return pc
}

func TestSessionMap_TTL(t *testing.T) {
	s := NewSessionMap(20 * time.Millisecond)
	inst := &Instance{URL: "http://a"}
	inst.healthy.Store(true)
	s.Set("k", inst)
	got, ok := s.Get("k")
	assert.True(t, ok)
	assert.Same(t, inst, got)

	time.Sleep(40 * time.Millisecond)
	_, ok = s.Get("k")
	assert.False(t, ok)
}

func TestSessionMap_Evict(t *testing.T) {
	s := NewSessionMap(10 * time.Millisecond)
	s.Set("k", &Instance{})
	time.Sleep(15 * time.Millisecond)
	assert.Equal(t, 1, s.Evict())
	assert.Equal(t, 0, s.Len())
}

func TestPool_PickPrefersLeastBusy(t *testing.T) {
	pc := poolConfig(t, "m", "http://a", "http://b")
	tr := &schema.LlamaCPPTranslator{}
	p, err := NewPool(pc, tr)
	assert.NoError(t, err)

	p.Instances[0].Acquire()
	p.Instances[0].Acquire()

	picked, err := p.Pick("s1")
	assert.NoError(t, err)
	assert.Equal(t, "http://b", picked.URL)
	p.Release(picked)
}

func TestPool_PickStickyWhenTied(t *testing.T) {
	pc := poolConfig(t, "m", "http://a", "http://b")
	tr := &schema.LlamaCPPTranslator{}
	p, err := NewPool(pc, tr)
	assert.NoError(t, err)

	p.Sessions.Set("k", p.Instances[1])

	picked, err := p.Pick("k")
	assert.NoError(t, err)
	assert.Equal(t, "http://b", picked.URL)
	p.Release(picked)
}

func TestPool_PickSkipsStickyWhenBusier(t *testing.T) {
	pc := poolConfig(t, "m", "http://a", "http://b")
	tr := &schema.LlamaCPPTranslator{}
	p, err := NewPool(pc, tr)
	assert.NoError(t, err)

	p.Sessions.Set("k", p.Instances[0])
	p.Instances[0].Acquire()

	picked, err := p.Pick("k")
	assert.NoError(t, err)
	assert.Equal(t, "http://b", picked.URL)
	p.Release(picked)
}

func TestPool_PickSkipsUnhealthy(t *testing.T) {
	pc := poolConfig(t, "m", "http://a", "http://b")
	tr := &schema.LlamaCPPTranslator{}
	p, err := NewPool(pc, tr)
	assert.NoError(t, err)

	p.Instances[0].SetHealthy(false)
	picked, err := p.Pick("k")
	assert.NoError(t, err)
	assert.Equal(t, "http://b", picked.URL)
	p.Release(picked)
}

func TestPool_PickReturnsErrorWhenAllUnhealthy(t *testing.T) {
	pc := poolConfig(t, "m", "http://a", "http://b")
	tr := &schema.LlamaCPPTranslator{}
	p, err := NewPool(pc, tr)
	assert.NoError(t, err)

	for _, inst := range p.Instances {
		inst.SetHealthy(false)
	}
	_, err = p.Pick("k")
	assert.ErrorIs(t, err, ErrNoHealthyInstance)
}

func TestRegistry_LookupByModel(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{Port: 1, APIKey: "x"},
		Pools: []config.PoolConfig{
			poolConfig(t, "m1", "http://a"),
			poolConfig(t, "m2", "http://b"),
		},
	}
	assert.NoError(t, cfg.Validate())
	reg, err := NewRegistry(cfg, schema.NewRegistry())
	assert.NoError(t, err)
	_, ok := reg.Get("m1")
	assert.True(t, ok)
	_, ok = reg.Get("missing")
	assert.False(t, ok)
	assert.Len(t, reg.All(), 2)
}

func TestHealthChecker_MarksUnhealthyAndRecovers(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(true)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !healthy.Load() {
			http.Error(w, "down", http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	pc := poolConfig(t, "m", srv.URL)
	pc.HealthCheckInterval = 1
	p, err := NewPool(pc, &schema.LlamaCPPTranslator{})
	assert.NoError(t, err)
	reg := &Registry{pools: map[string]*Pool{"m": p}}

	log := logging.NewNop()
	m := metrics.New()
	h := NewHealthChecker(reg, 10*time.Millisecond, log, m)

	assert.True(t, h.CheckOnce(context.Background(), p, p.Instances[0]))

	healthy.Store(false)
	assert.False(t, h.CheckOnce(context.Background(), p, p.Instances[0]))

	healthy.Store(true)
	assert.True(t, h.CheckOnce(context.Background(), p, p.Instances[0]))
}

func TestForwarder_InjectsBearerAndReturnsResponse(t *testing.T) {
	var seenAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer srv.Close()

	pc := poolConfig(t, "m", srv.URL)
	p, err := NewPool(pc, &schema.LlamaCPPTranslator{})
	assert.NoError(t, err)

	m := metrics.New()
	f := NewForwarder(logging.NewNop(), m)
	breq := &schema.BackendRequest{Method: http.MethodPost, Path: "/v1/chat/completions", Body: []byte(`{}`)}
	resp, finish, err := f.Do(context.Background(), p, p.Instances[0], breq)
	assert.NoError(t, err)
	defer resp.Body.Close()
	defer finish(nil)
	assert.Equal(t, "Bearer k", seenAuth)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestForwarder_FinishEmitsTokenFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer srv.Close()

	pc := poolConfig(t, "m", srv.URL)
	p, err := NewPool(pc, &schema.LlamaCPPTranslator{})
	assert.NoError(t, err)

	core, logs := observer.New(zap.InfoLevel)
	f := NewForwarder(zap.New(core), metrics.New())
	breq := &schema.BackendRequest{Method: http.MethodPost, Path: "/v1/chat/completions", Body: []byte(`{}`)}

	resp, finish, err := f.Do(context.Background(), p, p.Instances[0], breq)
	assert.NoError(t, err)
	_ = resp.Body.Close()
	finish(&schema.Usage{PromptTokens: 11, CompletionTokens: 5})

	entries := logs.FilterMessage("backend request").All()
	assert.Len(t, entries, 1)
	fields := entries[0].ContextMap()
	assert.Equal(t, int64(11), fields["prompt_tokens"])
	assert.Equal(t, int64(5), fields["completion_tokens"])
	_, hasTotal := fields["total_tokens"]
	assert.False(t, hasTotal)
}

func TestForwarder_FinishOmitsTokenFieldsWhenNilUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	pc := poolConfig(t, "m", srv.URL)
	p, err := NewPool(pc, &schema.LlamaCPPTranslator{})
	assert.NoError(t, err)

	core, logs := observer.New(zap.InfoLevel)
	f := NewForwarder(zap.New(core), metrics.New())
	breq := &schema.BackendRequest{Method: http.MethodGet, Path: "/health"}

	resp, finish, err := f.Do(context.Background(), p, p.Instances[0], breq)
	assert.NoError(t, err)
	_ = resp.Body.Close()
	finish(nil)

	entries := logs.FilterMessage("backend request").All()
	assert.Len(t, entries, 1)
	fields := entries[0].ContextMap()
	_, hasPrompt := fields["prompt_tokens"]
	_, hasComp := fields["completion_tokens"]
	assert.False(t, hasPrompt)
	assert.False(t, hasComp)
}

func TestHTTPStatusLabel(t *testing.T) {
	assert.Equal(t, "2xx", HTTPStatusLabel(200))
	assert.Equal(t, "4xx", HTTPStatusLabel(404))
	assert.Equal(t, "5xx", HTTPStatusLabel(503))
	assert.Equal(t, "3xx", HTTPStatusLabel(301))
	assert.Equal(t, "other", HTTPStatusLabel(0))
}
