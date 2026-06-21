import asyncio
import json
import os
import pytest
from unittest.mock import MagicMock, patch
from core.mcp_client import MCPClient, MCPHub

# --- Standard-Library-First Mock IO Stream Implementations ---

class MockStreamWriter:
    def __init__(self):
        self._queue = None
        self.written_data = []

    @property
    def queue(self):
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    def write(self, data: bytes):
        print(f"[MockStreamWriter] write called: {data!r}")
        self.written_data.append(data)
        self.queue.put_nowait(data)

    async def drain(self):
        print("[MockStreamWriter] drain called")
        await asyncio.sleep(0.001)

class MockStreamReader:
    def __init__(self):
        self._queue = None

    @property
    def queue(self):
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    async def readline(self) -> bytes:
        print("[MockStreamReader] readline called, waiting on queue...")
        res = await self.queue.get()
        print(f"[MockStreamReader] readline returned: {res!r}")
        return res

class MockProcess:
    def __init__(self):
        self.stdin = MockStreamWriter()
        self.stdout = MockStreamReader()
        self.stderr = MockStreamReader()
        self._wait_future = None

    def _get_wait_future(self):
        if self._wait_future is None:
            try:
                self._wait_future = asyncio.get_running_loop().create_future()
            except RuntimeError:
                pass
        return self._wait_future

    def kill(self):
        print("[MockProcess] kill called")
        fut = self._get_wait_future()
        if fut and not fut.done():
            fut.set_result(0)

    def terminate(self):
        print("[MockProcess] terminate called")
        fut = self._get_wait_future()
        if fut and not fut.done():
            fut.set_result(0)

    async def wait(self):
        print("[MockProcess] wait called")
        fut = self._get_wait_future()
        if fut is None:
            self._wait_future = asyncio.get_running_loop().create_future()
            fut = self._wait_future
        return await fut

@pytest.fixture
async def mock_subprocess(monkeypatch):
    """Fixture to patch asyncio.create_subprocess_exec to return a MockProcess."""
    proc = MockProcess()
    async def mock_exec(*args, **kwargs):
        print(f"[Mock subprocess] create_subprocess_exec called with args={args} kwargs={kwargs}")
        return proc
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec)
    return proc

async def run_rpc_responder(proc: MockProcess):
    """Background task to simulate standard JSON-RPC 2.0 responses from an MCP server."""
    print("[RPC Responder] Task started")
    try:
        while True:
            print("[RPC Responder] waiting on stdin queue...")
            data = await proc.stdin.queue.get()
            print(f"[RPC Responder] received from stdin: {data!r}")
            lines = data.decode("utf-8").strip().split("\n")
            for line in lines:
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[RPC Responder] JSONDecodeError: {e}")
                    continue
                
                req_id = req.get("id")
                method = req.get("method")
                print(f"[RPC Responder] Parsed request: req_id={req_id}, method={method}")
                
                if req_id is None:
                    # It's a JSON-RPC notification, do not reply
                    print("[RPC Responder] Notification received (no reply)")
                    continue
                
                if method == "initialize":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": {"name": "mock-subprocess-server", "version": "1.0.0"}
                        }
                    }
                elif method == "tools/list":
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "fake_tool",
                                    "description": "A fake tool in subprocess mode",
                                    "inputSchema": {"type": "object", "properties": {}}
                                }
                            ]
                        }
                    }
                elif method == "tools/call":
                    params = req.get("params", {})
                    tool_name = params.get("name")
                    args = params.get("arguments", {})
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Subprocess tool '{tool_name}' called with {json.dumps(args)}"
                                }
                            ]
                        }
                    }
                else:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown method: {method}"
                        }
                    }
                
                resp_bytes = json.dumps(resp).encode("utf-8") + b"\n"
                print(f"[RPC Responder] Sending response to stdout queue: {resp_bytes!r}")
                await proc.stdout.queue.put(resp_bytes)
    except asyncio.CancelledError:
        print("[RPC Responder] Task cancelled")
    except Exception as e:
        print(f"[RPC Responder] Task exception: {e}")


# --- Mock Mode Tests ---

@pytest.mark.asyncio
async def test_mcp_client_mock_sequential_thinking():
    """Test MCPClient mock mode with the sequential_thinking server."""
    client = MCPClient(name="sequential_thinking", command="fake", mock=True)
    assert client.name == "sequential_thinking"
    assert client.mock is True
    assert not client.connected

    # Establish mock connection
    connected = await client.connect()
    assert connected
    assert client.connected

    # Ensure re-connection is a no-op returning True
    assert await client.connect() is True

    # Check tools listing
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "thought"

    # Call 'thought' tool
    res = await client.call_tool("thought", {
        "thought": "Let us explore deep research details",
        "thoughtNumber": 1,
        "totalThoughts": 10,
        "nextThoughtNeeded": True
    })
    assert "content" in res
    assert "[Mock Thought 1/10]" in res["content"][0]["text"]
    assert len(client._mock_thoughts) == 1
    assert client._mock_thoughts[0]["thought"] == "Let us explore deep research details"

    await client.disconnect()
    assert not client.connected


@pytest.mark.asyncio
async def test_mcp_client_mock_knowledge_graph():
    """Test MCPClient mock mode with the knowledge_graph server."""
    client = MCPClient(name="knowledge_graph", command="fake", mock=True)
    await client.connect()

    # Check tools listing
    tools = await client.list_tools()
    names = {t["name"] for t in tools}
    assert "add_nodes" in names
    assert "add_relations" in names
    assert "search_graph" in names

    # Call 'add_nodes'
    nodes_res = await client.call_tool("add_nodes", {
        "nodes": [
            {"id": "n1", "label": "LangGraph"},
            {"id": "n2", "label": "StateGraph"}
        ]
    })
    assert "Successfully added 2 mock nodes." in nodes_res["content"][0]["text"]
    assert len(client._mock_graph_nodes) == 2

    # Call 'add_relations'
    rel_res = await client.call_tool("add_relations", {
        "relations": [
            {"source": "n1", "target": "n2", "type": "implements"}
        ]
    })
    assert "Successfully added 1 mock relationships." in rel_res["content"][0]["text"]
    assert len(client._mock_graph_edges) == 1

    # Call 'search_graph' with matching query
    search_res = await client.call_tool("search_graph", {"query": "StateGraph"})
    data = json.loads(search_res["content"][0]["text"])
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["id"] == "n2"
    assert len(data["relations"]) == 1

    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_mock_puppeteer():
    """Test MCPClient mock mode with the puppeteer/browser server."""
    client = MCPClient(name="puppeteer", command="fake", mock=True)
    await client.connect()

    # Check tools listing
    tools = await client.list_tools()
    names = {t["name"] for t in tools}
    assert "navigate" in names
    assert "get_html" in names
    assert "click" in names

    # Call 'navigate'
    nav_res = await client.call_tool("navigate", {"url": "https://openai.com"})
    assert "Successfully navigated to mock URL: https://openai.com" in nav_res["content"][0]["text"]
    assert client._mock_visited_urls["current"] == "https://openai.com"

    # Call 'get_html'
    html_res = await client.call_tool("get_html", {})
    assert "<html><body>Mock HTML content for https://openai.com" in html_res["content"][0]["text"]

    # Call 'click'
    click_res = await client.call_tool("click", {"selector": "button.submit"})
    assert "Successfully clicked mock selector 'button.submit'" in click_res["content"][0]["text"]

    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_mock_errors_and_edge_cases():
    """Test connection requirements and error handling in mock mode."""
    client = MCPClient(name="sequential_thinking", command="fake", mock=True)

    # 1. Calling list_tools before connect raises RuntimeError
    with pytest.raises(RuntimeError) as exc:
        await client.list_tools()
    assert "is not connected" in str(exc.value)

    # 2. Calling call_tool before connect raises RuntimeError
    with pytest.raises(RuntimeError) as exc:
        await client.call_tool("thought", {})
    assert "is not connected" in str(exc.value)

    await client.connect()

    # 3. Call unknown mock tool raises ValueError
    with pytest.raises(ValueError) as exc:
        await client.call_tool("invalid_tool", {})
    assert "Unknown mock tool 'invalid_tool'" in str(exc.value)

    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_mock_unsupported_name():
    """Test mock client with an unrecognized name returns no tools and fails on tool calls."""
    client = MCPClient(name="unknown_arbitrary_name", command="fake", mock=True)
    await client.connect()

    tools = await client.list_tools()
    assert tools == []

    with pytest.raises(ValueError) as exc:
        await client.call_tool("some_unsupported_tool", {})
    assert "Unknown mock tool 'some_unsupported_tool'" in str(exc.value)

    await client.disconnect()


# --- Subprocess Mode Tests (Using Mock IO Transport) ---

@pytest.mark.asyncio
async def test_mcp_client_subprocess_connection_and_flow(mock_subprocess):
    """Test successful connection handshake, tool listing, and tool calls in subprocess mode."""
    print("[Test] test_mcp_client_subprocess_connection_and_flow starting")
    resp_task = asyncio.create_task(run_rpc_responder(mock_subprocess))
    try:
        client = MCPClient(name="sub_proc", command="dummy_cmd", args=["--dummy"], mock=False)
        assert not client.connected

        # Establish connection with handshake
        print("[Test] Calling client.connect()")
        success = await client.connect()
        print(f"[Test] client.connect() returned {success}")
        assert success
        assert client.connected
        assert client.process is mock_subprocess

        # Handshake should have written connection requests
        assert len(mock_subprocess.stdin.written_data) >= 2

        # Verify list_tools over JSON-RPC 2.0
        print("[Test] Calling client.list_tools()")
        tools = await client.list_tools()
        print(f"[Test] client.list_tools() returned {tools}")
        assert len(tools) == 1
        assert tools[0]["name"] == "fake_tool"

        # Verify call_tool over JSON-RPC 2.0
        print("[Test] Calling client.call_tool()")
        res = await client.call_tool("fake_tool", {"query": "python rules"})
        print(f"[Test] client.call_tool() returned {res}")
        assert "content" in res
        assert "called with {\"query\": \"python rules\"}" in res["content"][0]["text"]

        print("[Test] Calling client.disconnect()")
        await client.disconnect()
        print("[Test] client.disconnect() completed")
        assert not client.connected
        assert client.process is None
    finally:
        print("[Test] Cancelling responder task")
        resp_task.cancel()
        try:
            await resp_task
        except asyncio.CancelledError:
            pass
        print("[Test] test_mcp_client_subprocess_connection_and_flow finished")


@pytest.mark.asyncio
async def test_mcp_client_subprocess_launch_failure(monkeypatch):
    """Test handling of subprocess execution failures (e.g. command not found)."""
    async def mock_failed_exec(*args, **kwargs):
        raise FileNotFoundError("Execution binary not found on PATH")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_failed_exec)

    client = MCPClient(name="failing_server", command="nonexistent", mock=False)
    success = await client.connect()
    assert not success
    assert not client.connected
    assert client.process is None


@pytest.mark.asyncio
async def test_mcp_client_subprocess_ignores_corrupt_json(mock_subprocess):
    """Test that the reader loop ignores corrupted, non-JSON output and continues working."""
    resp_task = asyncio.create_task(run_rpc_responder(mock_subprocess))
    try:
        client = MCPClient(name="corrupted_stream", command="dummy", mock=False)
        await client.connect()

        # Push non-JSON text to mock stdout stream
        await mock_subprocess.stdout.queue.put(b"Not JSON data\n")
        await mock_subprocess.stdout.queue.put(b"  \n") # Empty whitespace

        # Verify the client is still connected and operational
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "fake_tool"

        await client.disconnect()
    finally:
        resp_task.cancel()
        try:
            await resp_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_mcp_client_subprocess_handles_unexpected_eof(mock_subprocess):
    """Test that unexpected server EOF triggers clean connection shutdown."""
    resp_task = asyncio.create_task(run_rpc_responder(mock_subprocess))
    try:
        client = MCPClient(name="eof_stream", command="dummy", mock=False)
        await client.connect()

        # Trigger EOF on stdout
        await mock_subprocess.stdout.queue.put(b"")

        # Wait a brief moment for the read loop to catch EOF
        await asyncio.sleep(0.01)

        assert not client.connected
    finally:
        resp_task.cancel()
        try:
            await resp_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_mcp_client_subprocess_pending_requests_fail_on_eof(mock_subprocess):
    """Test that any blocked requests fail with RuntimeError if the server crashes/reaches EOF."""
    client = MCPClient(name="crash_stream", command="dummy", mock=False)
    
    # We simulate connection manually to avoid handshake blocking
    client.process = mock_subprocess
    client.connected = True
    client._read_task = asyncio.create_task(client._read_loop())

    # Invoke list_tools, which blocks awaiting response from stdout
    list_task = asyncio.create_task(client.list_tools())

    # Wait briefly for request to register in pending dict
    await asyncio.sleep(0.005)
    assert len(client._pending_requests) == 1

    # Send EOF to trigger read loop termination
    await mock_subprocess.stdout.queue.put(b"")

    with pytest.raises(RuntimeError) as exc:
        await list_task
    assert "MCP server read loop terminated" in str(exc.value)
    assert not client.connected
    assert len(client._pending_requests) == 0


@pytest.mark.asyncio
async def test_mcp_client_subprocess_handles_rpc_error(mock_subprocess):
    """Test that JSON-RPC errors returned from the server are raised correctly as RuntimeError."""
    resp_task = asyncio.create_task(run_rpc_responder(mock_subprocess))
    try:
        client = MCPClient(name="rpc_errors", command="dummy", mock=False)
        await client.connect()

        # Call a private request directly with an unknown method to trigger responder's RPC error handler
        with pytest.raises(RuntimeError) as exc:
            await client._send_request("unknown_unsupported_method", {})
        assert "Unknown method: unknown_unsupported_method" in str(exc.value)

        await client.disconnect()
    finally:
        resp_task.cancel()
        try:
            await resp_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_mcp_client_disconnect_resolves_pending_requests(mock_subprocess):
    """Test that active disconnect resolves any leftover pending requests with RuntimeError."""
    client = MCPClient(name="disconnect_midway", command="dummy", mock=False)
    client.process = mock_subprocess
    client.connected = True
    client._read_task = asyncio.create_task(client._read_loop())

    # Launch blockable call
    list_task = asyncio.create_task(client.list_tools())
    await asyncio.sleep(0.005)
    assert len(client._pending_requests) == 1

    # Call disconnect
    await client.disconnect()
    assert not client.connected
    assert len(client._pending_requests) == 0

    with pytest.raises(RuntimeError) as exc:
        await list_task
    assert any(msg in str(exc.value) for msg in ("MCP server read loop terminated", "Disconnected from MCP Server"))


# --- MCPHub Tests ---

def test_mcp_hub_initialization():
    """Test loading and initializing MCPHub with configuration variants."""
    custom_config = {
        "srv1": {"command": "cmd1", "mock": True},
        "srv2": {"command": "cmd2", "mock": False}
    }
    hub = MCPHub(servers_config=custom_config)
    assert hub.configs == custom_config
    assert hub.clients == {}

    # Test initialization fallback to default settings.MCP_SERVERS
    hub_default = MCPHub(servers_config=None)
    from config.settings import settings
    assert hub_default.configs == settings.MCP_SERVERS


@pytest.mark.asyncio
async def test_mcp_hub_connect_all_and_shutdown():
    """Test MCPHub connection and shutdown logic."""
    custom_config = {
        "srv1": {"command": "cmd1", "mock": True},
        "srv2": {"command": "cmd2", "mock": True}
    }
    hub = MCPHub(servers_config=custom_config)

    connect_statuses = await hub.connect_all()
    assert connect_statuses == {"srv1": True, "srv2": True}
    assert len(hub.clients) == 2
    assert hub.clients["srv1"].connected
    assert hub.clients["srv2"].connected

    await hub.shutdown()
    assert len(hub.clients) == 0


@pytest.mark.asyncio
async def test_mcp_hub_get_client_behaviors():
    """Test lazy-loading, retrieval, and error handling behaviors of MCPHub.get_client."""
    custom_config = {
        "srv1": {"command": "cmd1", "mock": True},
        "srv2": {"command": "cmd2", "mock": False}
    }
    hub = MCPHub(servers_config=custom_config)

    # 1. Lazy-connect when client is configured but not yet connected
    assert "srv1" not in hub.clients
    client = await hub.get_client("srv1")
    assert "srv1" in hub.clients
    assert client.connected

    # 2. Return existing client immediately on subsequent calls
    client_cached = await hub.get_client("srv1")
    assert client_cached is client

    # 3. Raise KeyError if retrieving a non-existent configuration
    with pytest.raises(KeyError) as exc:
        await hub.get_client("non_existent_server")
    assert "No MCP Server configuration found" in str(exc.value)

    # 4. Raise ConnectionError if lazy connection fails
    with patch.object(MCPClient, "connect", return_value=False):
        with pytest.raises(ConnectionError) as exc:
            await hub.get_client("srv2")
        assert "Could not connect to MCP Server 'srv2'" in str(exc.value)

    await hub.shutdown()


@pytest.mark.asyncio
async def test_mcp_hub_shutdown_ignores_disconnection_errors():
    """Test that MCPHub.shutdown succeeds even if one or more clients throw disconnect exceptions."""
    custom_config = {
        "srv1": {"command": "cmd1", "mock": True},
        "srv2": {"command": "cmd2", "mock": True}
    }
    hub = MCPHub(servers_config=custom_config)
    await hub.connect_all()

    # Stub client disconnect to raise an error
    async def failing_disconnect():
        raise RuntimeError("Disconnection failed unexpectedly")

    hub.clients["srv1"].disconnect = failing_disconnect

    # Shutdown should still clean up all clients and finalize successfully
    await hub.shutdown()
    assert len(hub.clients) == 0
