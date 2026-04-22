BINARY       ?= aiproxy
BIN_DIR      ?= bin
PKG          := ./...
LDFLAGS      := -s -w
IMAGE        ?= robcowart/aiproxy
IMAGE_TAG    ?= latest

.PHONY: build test docker

build:
	go build -ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(BINARY) ./cmd/aiproxy

test:
	go test -race -count=1 $(PKG)

docker:
	docker build -t $(IMAGE):$(IMAGE_TAG) .
