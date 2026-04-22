# syntax=docker/dockerfile:1.7

# Build aiproxy binary
FROM golang:1.25-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux \
    go build -trimpath -ldflags '-s -w' -o /out/aiproxy ./cmd/aiproxy

# Build aiproxy container
FROM alpine:latest

RUN apk add --no-cache ca-certificates tzdata \
 && addgroup -S aiproxy \
 && adduser -S -G aiproxy -H -D aiproxy \
 && mkdir -p /etc/aiproxy \
 && chown -R aiproxy:aiproxy /etc/aiproxy

COPY --from=builder /out/aiproxy /usr/local/bin/aiproxy

USER aiproxy
WORKDIR /etc/aiproxy

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/aiproxy"]
CMD ["--config", "/etc/aiproxy/config.yaml"]
