package frontend

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// requireAPIKey returns an http.Handler that enforces Bearer token auth against the configured proxy API key. Requests
// whose path is listed in pathsSkipped (e.g., "/health", "/metrics") bypass authentication.
func requireAPIKey(proxyKey string, pathsSkipped map[string]bool, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if pathsSkipped[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}
		h := r.Header.Get("Authorization")
		const prefix = "Bearer "
		if !strings.HasPrefix(h, prefix) {
			writeUnauthorized(w, "missing bearer token")
			return
		}
		tok := strings.TrimPrefix(h, prefix)
		if subtle.ConstantTimeCompare([]byte(tok), []byte(proxyKey)) != 1 {
			writeUnauthorized(w, "invalid api key")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeUnauthorized(w http.ResponseWriter, detail string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("WWW-Authenticate", `Bearer realm="aiproxy"`)
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"error":{"type":"invalid_api_key","message":"` + detail + `"}}`))
}
