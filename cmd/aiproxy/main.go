// Command aiproxy is a model-aware, OpenAI-compatible reverse proxy for pools of local/remote LLM backends.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/robcowart/aiproxy/pkg/backend"
	"github.com/robcowart/aiproxy/pkg/config"
	"github.com/robcowart/aiproxy/pkg/frontend"
	"github.com/robcowart/aiproxy/pkg/logging"
	"github.com/robcowart/aiproxy/pkg/metrics"
	"github.com/robcowart/aiproxy/pkg/schema"
	"github.com/spf13/pflag"
	"go.uber.org/zap"
)

func main() {
	var configPath string
	pflag.StringVarP(&configPath, "config", "c", "config.yaml", "path to the YAML configuration file")

	if err := rejectSingleDashLongFlags(os.Args[1:], pflag.CommandLine); err != nil {
		fmt.Fprintln(os.Stderr, "aiproxy:", err)
		os.Exit(2)
	}
	pflag.Parse()

	if err := run(configPath); err != nil {
		fmt.Fprintf(os.Stderr, "aiproxy: %v\n", err)
		os.Exit(1)
	}
}

func run(configPath string) error {
	cfg, err := config.Load(configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	log, err := logging.New(cfg.Server.LogLevel)
	if err != nil {
		return fmt.Errorf("build logger: %w", err)
	}
	defer func() { _ = log.Sync() }()

	log.Info("aiproxy starting", zap.String("config", configPath))
	cfg.LogEffective(log)

	m := metrics.New()

	pools, err := backend.NewRegistry(cfg, schema.NewRegistry())
	if err != nil {
		return fmt.Errorf("build pool registry: %w", err)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	hc := backend.NewHealthChecker(pools, minHealthInterval(cfg), log, m)
	hc.Start(ctx)
	pools.StartJanitors(ctx, 30*time.Second, m)

	fwd := backend.NewForwarder(log.Named("backend"), m)
	srv := frontend.NewFrontend(cfg, pools, fwd, log.Named("frontend"), m)

	if err := srv.ListenAndServe(ctx); err != nil {
		return err
	}
	log.Info("aiproxy stopped")
	return nil
}

// rejectSingleDashLongFlags enforces POSIX-style parsing: registered long flag names must be passed with a double dash
// (e.g. --config). Single-dash forms like -config would otherwise be silently parsed as the bundle -c onfig.
func rejectSingleDashLongFlags(args []string, fs *pflag.FlagSet) error {
	for _, a := range args {
		if a == "--" {
			return nil
		}
		if !strings.HasPrefix(a, "-") || strings.HasPrefix(a, "--") || len(a) < 3 {
			continue
		}
		name := strings.TrimPrefix(a, "-")
		if i := strings.IndexByte(name, '='); i >= 0 {
			name = name[:i]
		}
		if fs.Lookup(name) != nil {
			return fmt.Errorf("unknown flag %q (use --%s)", a, name)
		}
	}
	return nil
}

func minHealthInterval(cfg *config.Config) time.Duration {
	min := time.Hour
	for _, p := range cfg.Pools {
		if d := p.HealthCheckIntervalDuration(); d > 0 && d < min {
			min = d
		}
	}
	if min == time.Hour {
		return 30 * time.Second
	}
	return min
}
