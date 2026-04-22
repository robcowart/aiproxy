package backend

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/robcowart/aiproxy/pkg/config"
	"github.com/robcowart/aiproxy/pkg/metrics"
	"github.com/robcowart/aiproxy/pkg/schema"
)

// ErrNoHealthyInstance indicates that every instance in a pool is currently marked unhealthy.
var ErrNoHealthyInstance = errors.New("no healthy backend instance available")

// Pool groups a set of Instance objects that share the same Model Type (provider schema + endpoint + model name).
type Pool struct {
	Model      string
	Endpoint   config.EndpointType
	Schema     config.SchemaName
	Translator schema.Translator
	Instances  []*Instance
	HealthPath string
	Sessions   *SessionMap

	rrCounter atomic.Uint64
}

// NewPool constructs a Pool from its configuration plus a translator.
func NewPool(p config.PoolConfig, tr schema.Translator) (*Pool, error) {
	instances := make([]*Instance, 0, len(p.Instances))
	for _, ic := range p.Instances {
		inst, err := NewInstance(ic, 0)
		if err != nil {
			return nil, fmt.Errorf("pool %q: build instance %q: %w", p.Model, ic.URL, err)
		}
		instances = append(instances, inst)
	}
	hp := p.HealthPath
	if hp == "" {
		hp = tr.HealthPath()
	}
	return &Pool{
		Model:      p.Model,
		Endpoint:   p.Endpoint,
		Schema:     p.Schema,
		Translator: tr,
		Instances:  instances,
		HealthPath: hp,
		Sessions:   NewSessionMap(p.SessionTimeoutDuration()),
	}, nil
}

// Pick returns the best Instance to serve a request for sessionKey. It prefers any healthy instance whose current
// in-flight count equals the minimum across healthy instances; the sticky instance is used only when it is tied with
// the minimum (so a new request never has to wait for the sticky instance when another is free). The chosen instance is
// recorded in the session map and its Acquire() counter is incremented.
func (p *Pool) Pick(sessionKey string) (*Instance, error) {
	healthy := make([]*Instance, 0, len(p.Instances))
	for _, inst := range p.Instances {
		if inst.Healthy() {
			healthy = append(healthy, inst)
		}
	}
	if len(healthy) == 0 {
		return nil, ErrNoHealthyInstance
	}

	minLoad := int64(1<<62 - 1)
	for _, inst := range healthy {
		if n := inst.Inflight(); n < minLoad {
			minLoad = n
		}
	}

	var chosen *Instance
	if sticky, ok := p.Sessions.Get(sessionKey); ok && sticky.Healthy() && sticky.Inflight() == minLoad {
		chosen = sticky
	} else {
		candidates := healthy[:0]
		for _, inst := range healthy {
			if inst.Inflight() == minLoad {
				candidates = append(candidates, inst)
			}
		}
		idx := p.rrCounter.Add(1) % uint64(len(candidates))
		chosen = candidates[idx]
	}

	chosen.Acquire()
	p.Sessions.Set(sessionKey, chosen)
	return chosen, nil
}

// Release decrements the in-flight counter on the given instance.
func (p *Pool) Release(inst *Instance) {
	if inst != nil {
		inst.Release()
	}
}

// FirstHealthy returns the first healthy instance in the pool, or nil when none are healthy. Unlike Pick it does not
// modify in-flight counters or session mappings, so it is safe to use for out-of-band probes (e.g., fetching /v1/models
// metadata).
func (p *Pool) FirstHealthy() *Instance {
	for _, inst := range p.Instances {
		if inst.Healthy() {
			return inst
		}
	}
	return nil
}

// Registry is a lookup of pools by client-facing model name.
type Registry struct {
	pools map[string]*Pool
}

// NewRegistry builds a pool registry from the full configuration and a schema registry.
func NewRegistry(cfg *config.Config, schemas *schema.Registry) (*Registry, error) {
	r := &Registry{pools: make(map[string]*Pool, len(cfg.Pools))}
	for _, pc := range cfg.Pools {
		tr, err := schemas.Get(string(pc.Schema))
		if err != nil {
			return nil, fmt.Errorf("pool %q: %w", pc.Model, err)
		}
		p, err := NewPool(pc, tr)
		if err != nil {
			return nil, err
		}
		r.pools[p.Model] = p
	}
	return r, nil
}

// Get returns the pool serving the given model name.
func (r *Registry) Get(model string) (*Pool, bool) {
	p, ok := r.pools[model]
	return p, ok
}

// All returns every pool in the registry (unordered).
func (r *Registry) All() []*Pool {
	out := make([]*Pool, 0, len(r.pools))
	for _, p := range r.pools {
		out = append(out, p)
	}
	return out
}

// StartJanitors launches background goroutines that periodically evict expired session entries. The returned cancel
// stops them.
func (r *Registry) StartJanitors(ctx context.Context, interval time.Duration, m *metrics.Metrics) {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	for _, p := range r.All() {
		pool := p
		go func() {
			ticker := time.NewTicker(interval)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					pool.Sessions.Evict()
					if m != nil {
						m.SessionsActive.WithLabelValues(pool.Model).Set(float64(pool.Sessions.Len()))
					}
				}
			}
		}()
	}
}
