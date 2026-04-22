package backend

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/robcowart/aiproxy/pkg/metrics"
	"go.uber.org/zap"
)

// HealthChecker periodically probes each instance in each pool and updates their healthy flag, emitting metrics and zap
// log messages on transitions.
type HealthChecker struct {
	registry *Registry
	interval time.Duration
	timeout  time.Duration
	log      *zap.Logger
	metrics  *metrics.Metrics
}

// NewHealthChecker builds a HealthChecker.
func NewHealthChecker(reg *Registry, interval time.Duration, log *zap.Logger, m *metrics.Metrics) *HealthChecker {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	return &HealthChecker{
		registry: reg,
		interval: interval,
		timeout:  5 * time.Second,
		log:      log,
		metrics:  m,
	}
}

// Start launches a health-check goroutine per pool instance. It returns immediately; the goroutines stop when ctx is
// cancelled.
func (h *HealthChecker) Start(ctx context.Context) {
	for _, pool := range h.registry.All() {
		for _, inst := range pool.Instances {
			go h.run(ctx, pool, inst)
		}
	}
}

// CheckOnce probes a single instance once and returns whether it is healthy. Exposed primarily for tests.
func (h *HealthChecker) CheckOnce(ctx context.Context, pool *Pool, inst *Instance) bool {
	url := strings.TrimRight(inst.URL, "/") + pool.HealthPath
	reqCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	if inst.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+inst.APIKey)
	}
	resp, err := inst.Client().Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 500
}

func (h *HealthChecker) run(ctx context.Context, pool *Pool, inst *Instance) {
	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()

	check := func() {
		ok := h.CheckOnce(ctx, pool, inst)
		if ok {
			if inst.SetHealthy(true) {
				h.log.Info("backend recovered",
					zap.String("pool", pool.Model),
					zap.String("instance", inst.URL))
			}
			if h.metrics != nil {
				h.metrics.BackendUp.WithLabelValues(pool.Model, inst.URL).Set(1)
				h.metrics.HealthChecks.WithLabelValues(pool.Model, inst.URL, "ok").Inc()
			}
		} else {
			if inst.SetHealthy(false) {
				h.log.Warn("backend unhealthy",
					zap.String("pool", pool.Model),
					zap.String("instance", inst.URL))
			}
			if h.metrics != nil {
				h.metrics.BackendUp.WithLabelValues(pool.Model, inst.URL).Set(0)
				h.metrics.HealthChecks.WithLabelValues(pool.Model, inst.URL, "fail").Inc()
			}
		}
	}

	check()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			check()
		}
	}
}
