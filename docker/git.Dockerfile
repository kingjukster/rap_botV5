# Minimal image for git + git-lfs (push/pull with LFS)
FROM alpine:3.19

RUN apk add --no-cache git git-lfs

WORKDIR /app

# Default: show help
CMD ["git", "status"]
