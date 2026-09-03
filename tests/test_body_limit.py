"""Tests for body_limit.py - the ceiling on request body size.

The endpoints this guards are the ones that must parse a body before they
can know who sent it: the OAuth token endpoint, the two sign-in forms. So
what is checked here is that an oversized body is refused *without the
handler running*, by either route a client can take - a declared
Content-Length, or a chunked body that declares nothing.
"""

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from body_limit import BodySizeLimitMiddleware

LIMIT = 1024

_reached = []


async def echo(request):
    form = await request.form()
    _reached.append(len(form))
    return JSONResponse({"fields": len(form)})


@pytest.fixture(autouse=True)
def clear():
    _reached.clear()


@pytest.fixture
def client():
    app = Starlette(
        middleware=[Middleware(BodySizeLimitMiddleware, max_bytes=LIMIT,
                               exempt_paths=("/upload",))],
        routes=[
            Route("/form", endpoint=echo, methods=["GET", "POST"]),
            Route("/upload/file", endpoint=echo, methods=["POST"]),
        ],
    )
    return TestClient(app)


def _pairs(count):
    return "&".join(f"f{i}=1" for i in range(count))


def test_a_body_under_the_limit_is_passed_through(client):
    response = client.post("/form", content="a=1&b=2",
                           headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 200
    assert response.json() == {"fields": 2}


def test_a_declared_length_over_the_limit_is_refused_before_the_handler(client):
    body = _pairs(500)
    assert len(body) > LIMIT
    response = client.post("/form", content=body,
                           headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 413
    assert response.json() == {"error": "payload_too_large"}
    assert _reached == [], "the handler parsed a body it should never have seen"


def test_a_chunked_body_over_the_limit_is_refused_too(client):
    """Transfer-Encoding: chunked is not a way around the check.

    httpx sends a generator body chunked, with no Content-Length for the
    header check to catch - the bytes have to be counted as they arrive.
    """
    def chunks():
        for i in range(500):
            yield f"f{i}=1&".encode()

    response = client.post("/form", content=chunks(),
                           headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 413
    assert _reached == []


def test_a_chunked_body_under_the_limit_still_arrives_whole(client):
    def chunks():
        yield b"a=1&"
        yield b"b=2"

    response = client.post("/form", content=chunks(),
                           headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 200
    assert response.json() == {"fields": 2}


def test_a_get_is_not_touched(client):
    assert client.get("/form").status_code == 200


def test_an_exempt_path_keeps_its_own_limits(client):
    body = _pairs(500)
    response = client.post("/upload/file", content=body,
                           headers={"content-type": "application/x-www-form-urlencoded"})
    assert response.status_code == 200
