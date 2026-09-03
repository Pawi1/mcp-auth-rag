"""A ceiling on how large a request body this server will read.

Starlette honours max_fields and max_part_size while parsing a *multipart*
body and applies neither while parsing a *urlencoded* one: `request.form()`
on `application/x-www-form-urlencoded` builds a MultiDict out of however
many `a=1&a=1&...` pairs arrived, however many that turns out to be. The
parse is synchronous, so a single large body is both an allocation and a
stalled event loop, from a client that never had to authenticate - every
public form here (the OAuth token endpoint, the sign-in pages) has to parse
before it can know who is asking.

Refusing an oversized body before any handler asks for it closes that
whether or not the installed Starlette bounds urlencoded fields, and does
not depend on each handler remembering to check. Two ways in, because a
client chooses which:

  * Content-Length declared - refuse on the header, having read nothing.
  * Chunked, no length - count the bytes as they arrive and cut the stream
    off at the limit, so `Transfer-Encoding: chunked` is not a way around
    the check.

The limit is deliberately small. Every form this server serves carries a
handful of short fields; uploads travel on routes named in `exempt_paths`,
which keep their own limits.
"""

from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Methods that carry a body worth bounding. GET/HEAD/DELETE with a body is
# not something any client here does.
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

DEFAULT_MAX_BODY_BYTES = 256 * 1024


class BodySizeLimitMiddleware:
    """Refuse request bodies over `max_bytes` with 413.

    Written against the ASGI interface rather than BaseHTTPMiddleware: the
    point is to stop the body before anything downstream reads it, and
    BaseHTTPMiddleware would already have a Request built around it.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = DEFAULT_MAX_BODY_BYTES,
        exempt_paths: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _BODY_METHODS:
            return await self.app(scope, receive, send)

        # scope["path"] rather than the parsed URL: a path that does not
        # start with "/" moves the authority boundary when a URL is built by
        # concatenation, and the exemption list must match on what actually
        # routed.
        path = scope.get("path", "")
        if self.exempt_paths and path.startswith(self.exempt_paths):
            return await self.app(scope, receive, send)

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return await self._refuse(send)
            except ValueError:
                pass  # unparsable - fall through to counting what arrives

        read = 0
        over = False
        started = False   # the handler already began a response of its own
        answered = False  # the 413 has gone out

        async def counting_receive() -> Message:
            nonlocal read, over
            if over:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                read += len(message.get("body", b""))
                if read > self.max_bytes:
                    over = True
                    # Ending the stream is what stops the parser mid-body;
                    # the response the client sees is sent below.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started, answered
            if not over:
                if message["type"] == "http.response.start":
                    started = True
                return await send(message)
            # Past the limit. Anything the handler still had to say is
            # dropped - the body it was answering never finished arriving -
            # unless it had already begun answering, in which case there is
            # no second status line to send and the response stands.
            if started:
                return await send(message)
            if not answered:
                answered = True
                await self._refuse(send)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except ClientDisconnect:
            # Raised by the parser when the stream ends early, which above is
            # how the body was cut off. Anything else is a real disconnect.
            if not over:
                raise
        if over and not started and not answered:
            answered = True
            await self._refuse(send)

    async def _refuse(self, send: Send) -> None:
        body = b'{"error":"payload_too_large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
