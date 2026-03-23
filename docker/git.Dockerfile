# Minimal image for git + git-lfs + ssh (push/pull with LFS via SSH)
FROM alpine:3.19

RUN apk add --no-cache git git-lfs openssh-client \
  && mkdir -p /root/.ssh \
  && ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

WORKDIR /app

# Default: show help
CMD ["git", "status"]
