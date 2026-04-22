// Package backend owns the model-pool lifecycle: instances, health checks, sticky sessions, load-balanced selection,
// and HTTP forwarding.
package backend

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"sync/atomic"
	"time"

	"github.com/robcowart/aiproxy/pkg/config"
)

// Instance is a single backend endpoint inside a Pool.
type Instance struct {
	URL    string
	APIKey string

	healthy  atomic.Bool
	inflight atomic.Int64
	lastUsed atomic.Int64 // unix nanoseconds

	client *http.Client
}

// NewInstance constructs a Backend Instance from its configuration plus a timeout.
func NewInstance(cfg config.InstanceConfig, timeout time.Duration) (*Instance, error) {
	client, err := httpClientFor(cfg, timeout)
	if err != nil {
		return nil, err
	}
	inst := &Instance{
		URL:    cfg.URL,
		APIKey: cfg.APIKey,
		client: client,
	}
	inst.healthy.Store(true)
	return inst, nil
}

// Healthy returns whether the instance is currently considered healthy.
func (i *Instance) Healthy() bool { return i.healthy.Load() }

// SetHealthy toggles health state and returns whether it changed.
func (i *Instance) SetHealthy(v bool) bool {
	return i.healthy.Swap(v) != v
}

// Inflight returns the current in-flight request count.
func (i *Instance) Inflight() int64 { return i.inflight.Load() }

// Acquire increments the in-flight counter.
func (i *Instance) Acquire() { i.inflight.Add(1) }

// Release decrements the in-flight counter and touches last-used.
func (i *Instance) Release() {
	i.inflight.Add(-1)
	i.lastUsed.Store(time.Now().UnixNano())
}

// Client returns the instance's underlying HTTP client.
func (i *Instance) Client() *http.Client { return i.client }

func httpClientFor(cfg config.InstanceConfig, timeout time.Duration) (*http.Client, error) {
	tlsCfg := &tls.Config{InsecureSkipVerify: cfg.TLS.InsecureSkipVerify}
	if cfg.TLS.CAFile != "" {
		pem, err := os.ReadFile(cfg.TLS.CAFile)
		if err != nil {
			return nil, fmt.Errorf("read backend CA %q: %w", cfg.TLS.CAFile, err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("parse backend CA %q", cfg.TLS.CAFile)
		}
		tlsCfg.RootCAs = pool
	}
	tr := &http.Transport{
		TLSClientConfig:       tlsCfg,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   100,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 0,
		ExpectContinueTimeout: time.Second,
		ForceAttemptHTTP2:     true,
	}
	return &http.Client{
		Transport: tr,
		Timeout:   timeout,
	}, nil
}
