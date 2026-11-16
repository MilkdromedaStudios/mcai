"""Public join address: find the public IP and open the port on the home router with UPnP (stdlib only)."""
from __future__ import annotations

import logging
import re
import socket
import urllib.parse
import urllib.request

log = logging.getLogger("pyflayer.network")


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))      # no packet is sent; just picks the outgoing interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def public_ip(timeout: float = 6) -> str | None:
    for url in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                ip = r.read().decode().strip()
            if re.fullmatch(r"[0-9.]{7,15}|[0-9a-fA-F:]{3,45}", ip):
                return ip
        except Exception:  # noqa: BLE001
            continue
    return None


def _ssdp_location(timeout: float = 3) -> str | None:
    msg = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\n"
           "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n").encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.settimeout(timeout)
    try:
        s.sendto(msg, ("239.255.255.250", 1900))
        while True:
            data = s.recv(4096).decode(errors="replace")
            m = re.search(r"(?im)^location:\s*(\S+)", data)
            if m:
                return m.group(1)
    except OSError:
        return None
    finally:
        s.close()


def _control_url(location: str) -> tuple[str, str] | None:
    with urllib.request.urlopen(location, timeout=5) as r:
        xml = r.read().decode(errors="replace")
    for st in ("urn:schemas-upnp-org:service:WANIPConnection:1", "urn:schemas-upnp-org:service:WANIPConnection:2",
               "urn:schemas-upnp-org:service:WANPPPConnection:1"):
        i = xml.find(st)
        if i < 0:
            continue
        m = re.search(r"<controlURL>\s*([^<]+)\s*</controlURL>", xml[i:])
        if m:
            return urllib.parse.urljoin(location, m.group(1)), st
    return None


def upnp_forward(port: int, lan: str | None = None, description: str = "PyFlayer Minecraft") -> str:
    """Ask the router to forward TCP `port` to this machine. Returns a status message."""
    lan = lan or lan_ip()
    loc = _ssdp_location()
    if not loc:
        return "UPnP: no router answered (UPnP off?) - forward the port manually"
    try:
        found = _control_url(loc)
        if not found:
            return "UPnP: router has no port-mapping service - forward the port manually"
        url, st = found
        body = (f'<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
                f's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>'
                f'<u:AddPortMapping xmlns:u="{st}"><NewRemoteHost></NewRemoteHost>'
                f'<NewExternalPort>{port}</NewExternalPort><NewProtocol>TCP</NewProtocol>'
                f'<NewInternalPort>{port}</NewInternalPort><NewInternalClient>{lan}</NewInternalClient>'
                f'<NewEnabled>1</NewEnabled><NewPortMappingDescription>{description}</NewPortMappingDescription>'
                f'<NewLeaseDuration>0</NewLeaseDuration></u:AddPortMapping></s:Body></s:Envelope>').encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": 'text/xml; charset="utf-8"', "SOAPAction": f'"{st}#AddPortMapping"'})
        with urllib.request.urlopen(req, timeout=6):
            pass
        return f"UPnP: router now forwards TCP {port} -> {lan}:{port}"
    except Exception as e:  # noqa: BLE001
        return f"UPnP: router refused the mapping ({e}) - forward the port manually"


def port_open_locally(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False
