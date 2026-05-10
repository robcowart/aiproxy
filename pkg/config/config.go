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
	SchemaOllama    SchemaName = "ollama"
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
	Log    LogConfig    `koanf:"log"`
	Pools  []PoolConfig `koanf:"pools"`
}

// ServerConfig describes the client-facing HTTP server.
type ServerConfig struct {
	Host   string    `koanf:"host"`
	Port   int       `koanf:"port"`
	APIKey string    `koanf:"api_key"`
	TLS    TLSConfig `koanf:"tls"`
}

// LogConfig configures the process-wide logger.
type LogConfig struct {
	// Level is the minimum log level to emit. One of: debug, info, warn, error. Defaults to "info".
	Level string `koanf:"level"`
	// Format selects the encoder. One of: json (structured, production default), console (human-readable). Defaults
	// to "json".
	Format string `koanf:"format"`
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
	Parameters          map[string]any   `koanf:"parameters"`
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
	if c.Log.Level == "" {
		c.Log.Level = "info"
	}
	switch strings.ToLower(c.Log.Level) {
	case "debug", "info", "warn", "warning", "error":
	default:
		return fmt.Errorf("log.level: unknown level %q (expected debug|info|warn|error)", c.Log.Level)
	}
	if c.Log.Format == "" {
		c.Log.Format = "json"
	}
	switch strings.ToLower(c.Log.Format) {
	case "json", "console":
	default:
		return fmt.Errorf("log.format: unknown format %q (expected json|console)", c.Log.Format)
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
		case SchemaLlamaCPP, SchemaOpenAI, SchemaAnthropic, SchemaGoogle, SchemaOllama:
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
		if err := validatePoolParameters(i, p); err != nil {
			return err
		}
	}
	return nil
}

// paramKind enumerates the value type expected for a configured parameter override.
type paramKind int

const (
	paramKindNumber paramKind = iota
	paramKindInt
	paramKindBool
	paramKindString
	paramKindStringArray
	paramKindObject
)

type paramSpec struct {
	kind         paramKind
	llamaCPPOnly bool
}

// chatParamSpecs is the allow-list of parameter override keys for chat_completions pools. Entries with
// llamaCPPOnly=true may only be used when the pool's schema is SchemaLlamaCPP.
var chatParamSpecs = map[string]paramSpec{
	"max_tokens":        {kind: paramKindInt},
	"temperature":       {kind: paramKindNumber},
	"top_p":             {kind: paramKindNumber},
	"top_k":             {kind: paramKindInt},
	"min_p":             {kind: paramKindNumber},
	"stream":            {kind: paramKindBool},
	"stop":              {kind: paramKindStringArray},
	"presence_penalty":  {kind: paramKindNumber},
	"frequency_penalty": {kind: paramKindNumber},
	"repeat_penalty":    {kind: paramKindNumber},
	"n":                 {kind: paramKindInt},
	"seed":              {kind: paramKindInt},
	"logprobs":          {kind: paramKindInt},
	"echo":              {kind: paramKindBool},
	"suffix":            {kind: paramKindString},

	"mirostat":     {kind: paramKindInt, llamaCPPOnly: true},
	"mirostat_tau": {kind: paramKindNumber, llamaCPPOnly: true},
	"mirostat_eta": {kind: paramKindNumber, llamaCPPOnly: true},
	"grammar":      {kind: paramKindString, llamaCPPOnly: true},
	"json_schema":  {kind: paramKindObject, llamaCPPOnly: true},
	"cache_prompt": {kind: paramKindBool, llamaCPPOnly: true},
}

func validatePoolParameters(i int, p *PoolConfig) error {
	if len(p.Parameters) == 0 {
		return nil
	}
	if p.Endpoint != EndpointChatCompletions {
		return fmt.Errorf("pools[%d] (%s): parameters are only supported on chat_completions pools", i, p.Model)
	}
	for name, val := range p.Parameters {
		spec, ok := chatParamSpecs[name]
		if !ok {
			return fmt.Errorf("pools[%d] (%s): unknown parameter %q", i, p.Model, name)
		}
		if spec.llamaCPPOnly && p.Schema != SchemaLlamaCPP {
			return fmt.Errorf("pools[%d] (%s): parameter %q is only supported with the llamacpp schema", i, p.Model, name)
		}
		if err := checkParamType(name, val, spec.kind); err != nil {
			return fmt.Errorf("pools[%d] (%s): %w", i, p.Model, err)
		}
	}
	return nil
}

func checkParamType(name string, val any, kind paramKind) error {
	switch kind {
	case paramKindNumber:
		switch val.(type) {
		case int, int32, int64, float32, float64:
			return nil
		}
		return fmt.Errorf("parameter %q: expected number, got %T", name, val)
	case paramKindInt:
		switch v := val.(type) {
		case int, int32, int64:
			return nil
		case float32:
			if float32(int64(v)) == v {
				return nil
			}
		case float64:
			if float64(int64(v)) == v {
				return nil
			}
		}
		return fmt.Errorf("parameter %q: expected integer, got %T", name, val)
	case paramKindBool:
		if _, ok := val.(bool); ok {
			return nil
		}
		return fmt.Errorf("parameter %q: expected boolean, got %T", name, val)
	case paramKindString:
		if _, ok := val.(string); ok {
			return nil
		}
		return fmt.Errorf("parameter %q: expected string, got %T", name, val)
	case paramKindStringArray:
		arr, ok := val.([]any)
		if !ok {
			return fmt.Errorf("parameter %q: expected array of strings, got %T", name, val)
		}
		for idx, item := range arr {
			if _, ok := item.(string); !ok {
				return fmt.Errorf("parameter %q[%d]: expected string, got %T", name, idx, item)
			}
		}
		return nil
	case paramKindObject:
		if _, ok := val.(map[string]any); ok {
			return nil
		}
		return fmt.Errorf("parameter %q: expected object, got %T", name, val)
	}
	return fmt.Errorf("parameter %q: unsupported kind", name)
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
			"parameters":                p.Parameters,
			"instances":                 insts,
		})
	}
	log.Info("effective configuration",
		zap.Any("server", map[string]any{
			"host":    c.Server.Host,
			"port":    c.Server.Port,
			"api_key": redact(c.Server.APIKey),
			"tls":     c.Server.TLS,
		}),
		zap.Any("log", map[string]any{
			"level":  c.Log.Level,
			"format": c.Log.Format,
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
