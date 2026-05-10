// Package logging provides zap logger construction and shared request-logging helpers used by the frontend and backend
// packages.
package logging

import (
	"fmt"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// New constructs a zap logger with the provided level and encoder format. Accepted levels (case-insensitive): debug,
// info, warn, error. Accepted formats (case-insensitive): json (structured production default), console
// (human-readable). An empty format defaults to json.
func New(level, format string) (*zap.Logger, error) {
	lvl, err := parseLevel(level)
	if err != nil {
		return nil, err
	}
	enc, err := parseFormat(format)
	if err != nil {
		return nil, err
	}

	cfg := zap.NewProductionConfig()
	cfg.Encoding = enc
	cfg.Level = zap.NewAtomicLevelAt(lvl)
	cfg.DisableStacktrace = true
	cfg.EncoderConfig.TimeKey = "ts"
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	cfg.EncoderConfig.EncodeDuration = zapcore.StringDurationEncoder
	if enc == "console" {
		cfg.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	}

	return cfg.Build()
}

// NewNop returns a no-op logger, useful for tests.
func NewNop() *zap.Logger { return zap.NewNop() }

func parseLevel(s string) (zapcore.Level, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "info":
		return zapcore.InfoLevel, nil
	case "debug":
		return zapcore.DebugLevel, nil
	case "warn", "warning":
		return zapcore.WarnLevel, nil
	case "error":
		return zapcore.ErrorLevel, nil
	default:
		return zapcore.InfoLevel, fmt.Errorf("unknown log level %q", s)
	}
}

func parseFormat(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "json":
		return "json", nil
	case "console":
		return "console", nil
	default:
		return "", fmt.Errorf("unknown log format %q (expected json|console)", s)
	}
}
