"""Synthetic tests may contact loopback fixtures, never ambient paid providers."""
import ipaddress
import socket

import pytest


def pytest_addoption(parser):
    parser.addoption('--allow-remote-inference', action='store_true', default=False,
        help='Explicit opt-in for separately configured real-provider reproduction')


@pytest.fixture(autouse=True)
def local_fixture_network_only(monkeypatch, request):
    if request.config.getoption('--allow-remote-inference'):
        return
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def check(host):
        if host is None or host in ('localhost', b'localhost'):
            return
        try:
            value = host.decode() if isinstance(host, bytes) else host
            if ipaddress.ip_address(value).is_loopback:
                return
        except (ValueError, TypeError):
            pass
        raise OSError('Remote network disabled in synthetic tests; use owned loopback fixtures')

    def resolve(host, *args, **kwargs):
        check(host)
        return original_resolve(host, *args, **kwargs)

    def connect(sock, address):
        if isinstance(address, tuple):
            check(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if isinstance(address, tuple):
            check(address[0])
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, 'getaddrinfo', resolve)
    monkeypatch.setattr(socket.socket, 'connect', connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', connect_ex)
