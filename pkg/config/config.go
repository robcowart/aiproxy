// Package config loads, validates, and reports the aiproxy runtime configuration using koanf.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/knadh/koanf/parsers/yaml"
	envprov "github.com/knadh/koanf/providers/env/v2"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	"go.uber.org/zap"
)

// EnvPrefix is the prefix used when reading environment overrides.
const EnvPrefix = "AIPROXY_"

// SchemaName identifies a backend provider translator.
type SchemaName string

const (
	SchemaLlamaCPP  SchemaName = "llamacpp"
	SchemaOpenAI    SchemaName = "openai"
	SchemaAnthropic SchemaName = "anthropic"
	SchemaGoogle    SchemaName = "google"
)

// EndpointType identifies which OpenAI-compatible route a pool serves.
type EndpointType string

const (
	EndpointChatCompletions EndpointType = "chat_completions"
	EndpointEmbeddings      EndpointType = "embeddings"
	EndpointRerank          EndpointType = "rerank"
)

// Config is the fully validated runtime configuration.
type Config struct {
	Server ServerConfig `koanf:"server"`
	Pools  []PoolConfig `koanf:"pools"`
}

// ServerConfig describes the client-facing HTTP server.
type ServerConfig struct {
	Host     string    `koanf:"host"`
	Port     int       `koanf:"port"`
	APIKey   string    `koanf:"api_key"`
	LogLevel string    `koanf:"log_level"`
	TLS      TLSConfig `koanf:"tls"`
}

// TLSConfig configures optional HTTPS for the client-facing server.
type TLSConfig struct {
	Enabled  bool   `koanf:"enabled"`
	CertFile string `koanf:"cert_file"`
	KeyFile  string `koanf:"key_file"`
}

// PoolConfig describes a single Model Pool (all instances share a provider schema, a routing model name, and an
// endpoint type).
type PoolConfig struct {
	Model               string           `koanf:"model"`
	Endpoint            EndpointType     `koanf:"endpoint"`
	Schema              SchemaName       `koanf:"schema"`
	Instances           []InstanceConfig `koanf:"instances"`
	SessionTimeout      int              `koanf:"session_timeout"`
	HealthCheckInterval int              `koanf:"health_check_interval"`
	HealthPath          string           `koanf:"health_path"`
	MaxInflight         int              `koanf:"max_inflight_per_instance"`
	RateLimit           RateLimitConfig  `koanf:"rate_limit"`
}

// RateLimitConfig is optional per-pool token-bucket configuration.
type RateLimitConfig struct {
	RPS   float64 `koanf:"rps"`
	Burst int     `koanf:"burst"`
}

// InstanceConfig describes a single backend Model Instance.
type InstanceConfig struct {
	URL    string            `koanf:"url"`
	APIKey string            `koanf:"api_key"`
	TLS    InstanceTLSConfig `koanf:"tls"`
}

// InstanceTLSConfig is optional per-instance TLS configuration.
type InstanceTLSConfig struct {
	CAFile             string `koanf:"ca_file"`
	InsecureSkipVerify bool   `koanf:"insecure_skip_verify"`
}

// SessionTimeoutDuration returns the pool session timeout as a duration.
func (p PoolConfig) SessionTimeoutDuration() time.Duration {
	return time.Duration(p.SessionTimeout) * time.Second
}

// HealthCheckIntervalDuration returns the health-check cadence as a duration.
func (p PoolConfig) HealthCheckIntervalDuration() time.Duration {
	return time.Duration(p.HealthCheckInterval) * time.Second
}

// Load reads configuration from the given YAML file path, then applies any AIPROXY_* environment overrides, validates
// the result, and returns it.
func Load(path string) (*Config, error) {
	k := koanf.New(".")

	if err := k.Load(file.Provider(path), yaml.Parser()); err != nil {
		return nil, fmt.Errorf("load config file %q: %w", path, err)
	}

	envProv := envprov.Provider(".", envprov.Opt{
		Prefix: EnvPrefix,
		TransformFunc: func(key, value string) (string, any) {
			key = strings.TrimPrefix(key, EnvPrefix)
			key = strings.ReplaceAll(strings.ToLower(key), "__", ".")
			return key, value
		},
	})
	if err := k.Load(envProv, nil); err != nil {
		return nil, fmt.Errorf("load env overrides: %w", err)
	}

	var cfg Config
	unmarshalConf := koanf.UnmarshalConf{Tag: "koanf"}
	if err := k.UnmarshalWithConf("", &cfg, unmarshalConf); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// Validate enforces required fields and consistency rules.
func (c *Config) Validate() error {
	if c.Server.Host == "" {
		c.Server.Host = "0.0.0.0"
	}
	if c.Server.Port == 0 {
		c.Server.Port = 8080
	}
	if c.Server.APIKey == "" {
		return errors.New("server.api_key is required")
	}
	if c.Server.LogLevel == "" {
		c.Server.LogLevel = "info"
	}
	if c.Server.TLS.Enabled {
		if c.Server.TLS.CertFile == "" || c.Server.TLS.KeyFile == "" {
			return errors.New("server.tls.cert_file and server.tls.key_file are required when tls.enabled is true")
		}
	}
	if len(c.Pools) == 0 {
		return errors.New("at least one pool must be configured")
	}
	seen := make(map[string]struct{}, len(c.Pools))
	for i := range c.Pools {
		p := &c.Pools[i]
		if p.Model == "" {
			return fmt.Errorf("pools[%d].model is required", i)
		}
		if _, dup := seen[p.Model]; dup {
			return fmt.Errorf("duplicate pool model name %q", p.Model)
		}
		seen[p.Model] = struct{}{}

		switch p.Endpoint {
		case EndpointChatCompletions, EndpointEmbeddings, EndpointRerank:
		default:
			return fmt.Errorf("pools[%d] (%s): invalid endpoint %q", i, p.Model, p.Endpoint)
		}
		switch p.Schema {
		case SchemaLlamaCPP, SchemaOpenAI, SchemaAnthropic, SchemaGoogle:
		default:
			return fmt.Errorf("pools[%d] (%s): invalid schema %q", i, p.Model, p.Schema)
		}
		if len(p.Instances) == 0 {
			return fmt.Errorf("pools[%d] (%s): at least one instance is required", i, p.Model)
		}
		if p.SessionTimeout < 0 {
			return fmt.Errorf("pools[%d] (%s): session_timeout must be >= 0", i, p.Model)
		}
		if p.HealthCheckInterval <= 0 {
			p.HealthCheckInterval = 30
		}
		for j, inst := range p.Instances {
			if inst.URL == "" {
				return fmt.Errorf("pools[%d].instances[%d]: url is required", i, j)
			}
			u, err := url.Parse(inst.URL)
			if err != nil || (u.Scheme != "http" && u.Scheme != "https") {
				return fmt.Errorf("pools[%d].instances[%d]: invalid url %q", i, j, inst.URL)
			}
			if inst.APIKey == "" {
				return fmt.Errorf("pools[%d].instances[%d]: api_key is required", i, j)
			}
		}
	}
	return nil
}

// LogEffective writes the effective configuration to the logger at info level, redacting all API keys.
func (c *Config) LogEffective(log *zap.Logger) {
	redactedPools := make([]any, 0, len(c.Pools))
	for _, p := range c.Pools {
		insts := make([]map[string]any, 0, len(p.Instances))
		for _, i := range p.Instances {
			insts = append(insts, map[string]any{
				"url":     i.URL,
				"api_key": redact(i.APIKey),
				"tls":     i.TLS,
			})
		}
		redactedPools = append(redactedPools, map[string]any{
			"model":                     p.Model,
			"endpoint":                  p.Endpoint,
			"schema":                    p.Schema,
			"session_timeout":           p.SessionTimeout,
			"health_check_interval":     p.HealthCheckInterval,
			"health_path":               p.HealthPath,
			"max_inflight_per_instance": p.MaxInflight,
			"rate_limit":                p.RateLimit,
			"instances":                 insts,
		})
	}
	log.Info("effective configuration",
		zap.Any("server", map[string]any{
			"host":      c.Server.Host,
			"port":      c.Server.Port,
			"api_key":   redact(c.Server.APIKey),
			"log_level": c.Server.LogLevel,
			"tls":       c.Server.TLS,
		}),
		zap.Any("pools", redactedPools),
	)
}

// redact masks every character of a secret except the final four.
func redact(s string) string {
	if s == "" {
		return ""
	}
	if len(s) <= 4 {
		return "****"
	}
	return strings.Repeat("*", len(s)-4) + s[len(s)-4:]
}
