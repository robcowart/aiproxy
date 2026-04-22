package backend

import (
	"sync"
	"time"
)

// sessionEntry maps a session key to an instance plus a TTL expiration.
type sessionEntry struct {
	instance  *Instance
	expiresAt time.Time
}

// SessionMap is a TTL-bounded map from session key to Instance. All operations are safe for concurrent use; expired
// entries are evicted lazily on access and periodically by the Janitor goroutine.
type SessionMap struct {
	mu      sync.Mutex
	entries map[string]sessionEntry
	ttl     time.Duration
}

// NewSessionMap returns a SessionMap with the given TTL.
func NewSessionMap(ttl time.Duration) *SessionMap {
	return &SessionMap{
		entries: make(map[string]sessionEntry),
		ttl:     ttl,
	}
}

// Get returns the instance bound to key if it exists and has not expired.
func (s *SessionMap) Get(key string) (*Instance, bool) {
	if s.ttl <= 0 {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[key]
	if !ok {
		return nil, false
	}
	if time.Now().After(e.expiresAt) {
		delete(s.entries, key)
		return nil, false
	}
	return e.instance, true
}

// Set binds key -> inst with a fresh TTL.
func (s *SessionMap) Set(key string, inst *Instance) {
	if s.ttl <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[key] = sessionEntry{instance: inst, expiresAt: time.Now().Add(s.ttl)}
}

// Len returns the current number of entries (including possibly expired ones not yet evicted).
func (s *SessionMap) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}

// Evict removes entries that have passed their TTL and returns the number removed.
func (s *SessionMap) Evict() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	removed := 0
	for k, e := range s.entries {
		if now.After(e.expiresAt) {
			delete(s.entries, k)
			removed++
		}
	}
	return removed
}
