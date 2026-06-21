import asyncio
import json
import os
import structlog
from typing import Any, Dict, List, Optional, Union

logger = structlog.get_logger("deep-research")

class MCPClient:
    """A generic python-native Model Context Protocol (MCP) Client over stdio transport.
    Supports JSON-RPC 2.0, handshakes, and provides robust offline mocking capability.
    """
    
    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        mock: bool = False
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or os.environ.copy()
        self.mock = mock
        
        self.process: Optional[asyncio.subprocess.Process] = None
        self._read_task: Optional[asyncio.Task] = None
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._request_id = 1
        self.connected = False
        
        # In-memory store for mock knowledge graph / state
        self._mock_graph_nodes = []
        self._mock_graph_edges = []
        self._mock_thoughts = []
        self._mock_visited_urls = {}

    async def connect(self) -> bool:
        """Establishes connection to the MCP Server and performs initial handshake."""
        if self.connected:
            return True
            
        logger.info("Connecting to MCP server", server_name=self.name, mock=self.mock)
        
        if self.mock:
            self.connected = True
            logger.info("Successfully connected to mock MCP server", server_name=self.name)
            return True
            
        try:
            # Spawn the server subprocess
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env
            )
            
            self.connected = True
            # Start background reader
            self._read_task = asyncio.create_task(self._read_loop())
            
            # Execute initialization handshake
            init_response = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "deep-research-client",
                        "version": "0.1.0"
                    }
                }
            )
            
            logger.debug("MCP server initialized", server_name=self.name, response=init_response)
            
            # Send initialized notification (no ID)
            await self._send_notification("notifications/initialized", {})
            
            logger.info("Handshake complete. Connected to MCP server", server_name=self.name)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to MCP server", server_name=self.name, error=str(e))
            self.connected = False
            if self.process:
                try:
                    self.process.kill()
                except Exception:
                    pass
                self.process = None
            return False

    async def disconnect(self):
        """Cleanly disconnects and terminates the MCP Server process."""
        if not self.connected:
            return
            
        logger.info("Disconnecting from MCP server", server_name=self.name)
        
        if not self.mock:
            if self._read_task:
                self._read_task.cancel()
                try:
                    await self._read_task
                except asyncio.CancelledError:
                    pass
                self._read_task = None
                
            if self.process:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                self.process = None
                
        self.connected = False
        
        # Resolve any leftover pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError("Disconnected from MCP Server"))
        self._pending_requests.clear()
        
        logger.info("Disconnected from MCP server", server_name=self.name)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Queries the server for its available tools."""
        if not self.connected:
            raise RuntimeError(f"MCP Client '{self.name}' is not connected.")
            
        if self.mock:
            return self._get_mock_tools()
            
        response = await self._send_request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> Dict[str, Any]:
        """Calls a specific tool on the MCP Server."""
        if not self.connected:
            raise RuntimeError(f"MCP Client '{self.name}' is not connected.")
            
        if self.mock:
            return await self._execute_mock_tool(name, arguments)
            
        return await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

    # Internal JSON-RPC helpers
    
    async def _send_request(self, method: str, params: dict) -> Dict[str, Any]:
        req_id = self._request_id
        self._request_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }
        
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future
        
        await self._write_line(json.dumps(request))
        
        try:
            return await future
        except Exception as e:
            logger.error("Error executing request", method=method, req_id=req_id, error=str(e))
            raise

    async def _send_notification(self, method: str, params: dict):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        await self._write_line(json.dumps(notification))

    async def _write_line(self, data: str):
        if not self.process or not self.process.stdin:
            raise RuntimeError("Subprocess stdin is not available.")
            
        try:
            self.process.stdin.write(f"{data}\n".encode("utf-8"))
            await self.process.stdin.drain()
        except Exception as e:
            logger.error("Failed to write to stdin", server_name=self.name, error=str(e))
            raise

    async def _read_loop(self):
        """Background loop reading JSON-RPC messages from stdout."""
        if not self.process or not self.process.stdout:
            return
            
        while self.connected:
            try:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    logger.warning("MCP stdout EOF reached", server_name=self.name)
                    break
                    
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                    
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Ignored non-JSON line from MCP server", server_name=self.name, line=line)
                    continue
                    
                # Check if it's a response
                if "id" in message:
                    req_id = message["id"]
                    future = self._pending_requests.pop(req_id, None)
                    if future and not future.done():
                        if "error" in message:
                            future.set_exception(RuntimeError(message["error"]))
                        else:
                            future.set_result(message.get("result", {}))
                else:
                    # Request/Notification from server (not currently handled)
                    logger.debug("Received request/notification from server", message=message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in MCP read loop", server_name=self.name, error=str(e))
                break
                
        # Handle unexpected crash/disconnection
        self.connected = False
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError("MCP server read loop terminated."))
        self._pending_requests.clear()

    # Mock Server Logic

    def _get_mock_tools(self) -> List[Dict[str, Any]]:
        """Returns mock tool definitions depending on the server name."""
        if "sequential_thinking" in self.name:
            return [
                {
                    "name": "thought",
                    "description": "Log a logical thinking step to plan, revise, and verify reasoning.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "thought": {"type": "string"},
                            "thoughtNumber": {"type": "integer"},
                            "totalThoughts": {"type": "integer"},
                            "nextThoughtNeeded": {"type": "boolean"}
                        },
                        "required": ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"]
                    }
                }
            ]
        elif "knowledge_graph" in self.name:
            return [
                {
                    "name": "add_nodes",
                    "description": "Add semantic nodes to the research knowledge graph.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "nodes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "label": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                {
                    "name": "add_relations",
                    "description": "Add relationships between semantic nodes.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "type": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                {
                    "name": "search_graph",
                    "description": "Query the semantic knowledge graph.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                }
            ]
        elif "puppeteer" in self.name or "browser" in self.name:
            return [
                {
                    "name": "navigate",
                    "description": "Navigate the headless browser to a specific URL.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"}
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "get_html",
                    "description": "Get the HTML content of the current page.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "click",
                    "description": "Click a element matching a selector on the current page.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"}
                        },
                        "required": ["selector"]
                    }
                }
            ]
        return []

    async def _execute_mock_tool(self, name: str, arguments: dict) -> Dict[str, Any]:
        """Simulates the execution of mock tools for testing/offline support."""
        await asyncio.sleep(0.01) # Simulate minor network/processing latency
        
        if name == "thought":
            thought = arguments.get("thought", "")
            num = arguments.get("thoughtNumber", 1)
            total = arguments.get("totalThoughts", 10)
            self._mock_thoughts.append(arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"[Mock Thought {num}/{total}]: {thought} (Logged successfully)"
                    }
                ]
            }
            
        elif name == "add_nodes":
            nodes = arguments.get("nodes", [])
            self._mock_graph_nodes.extend(nodes)
            return {
                "content": [{"type": "text", "text": f"Successfully added {len(nodes)} mock nodes."}]
            }
            
        elif name == "add_relations":
            relations = arguments.get("relations", [])
            self._mock_graph_edges.extend(relations)
            return {
                "content": [{"type": "text", "text": f"Successfully added {len(relations)} mock relationships."}]
            }
            
        elif name == "search_graph":
            q = arguments.get("query", "")
            matches = [n for n in self._mock_graph_nodes if q.lower() in n.get("label", "").lower() or q.lower() in n.get("id", "").lower()]
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"nodes": matches, "relations": self._mock_graph_edges})
                    }
                ]
            }
            
        elif name == "navigate":
            url = arguments.get("url", "")
            self._mock_visited_urls["current"] = url
            return {
                "content": [{"type": "text", "text": f"Successfully navigated to mock URL: {url}"}]
            }
            
        elif name == "get_html":
            curr_url = self._mock_visited_urls.get("current", "about:blank")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"<html><body>Mock HTML content for {curr_url} with deep research evidence.</body></html>"
                    }
                ]
            }
            
        elif name == "click":
            sel = arguments.get("selector", "")
            return {
                "content": [{"type": "text", "text": f"Successfully clicked mock selector '{sel}'"}]
            }
            
        raise ValueError(f"Unknown mock tool '{name}' on client '{self.name}'")


class MCPHub:
    """A central gateway or manager for all configured MCP servers."""
    
    def __init__(self, servers_config: Optional[Dict[str, dict]] = None):
        # Fallback to configured settings if none passed
        if servers_config is None:
            from config.settings import settings
            servers_config = settings.MCP_SERVERS
            
        self.configs = servers_config or {}
        self.clients: Dict[str, MCPClient] = {}

    async def connect_all(self) -> Dict[str, bool]:
        """Connects to all configured MCP servers."""
        results = {}
        for server_name, cfg in self.configs.items():
            client = MCPClient(
                name=server_name,
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env"),
                mock=cfg.get("mock", False)
            )
            success = await client.connect()
            if success:
                self.clients[server_name] = client
            results[server_name] = success
        return results

    async def get_client(self, name: str) -> MCPClient:
        """Retrieves an active connected client. If not connected, attempts connection."""
        if name in self.clients and self.clients[name].connected:
            return self.clients[name]
            
        cfg = self.configs.get(name)
        if not cfg:
            raise KeyError(f"No MCP Server configuration found for '{name}'")
            
        client = MCPClient(
            name=name,
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            env=cfg.get("env"),
            mock=cfg.get("mock", False)
        )
        success = await client.connect()
        if not success:
            raise ConnectionError(f"Could not connect to MCP Server '{name}'")
            
        self.clients[name] = client
        return client

    async def shutdown(self):
        """Disconnects all active clients."""
        disconnect_tasks = [client.disconnect() for client in self.clients.values()]
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        self.clients.clear()
        logger.info("All MCP Hub clients shut down.")
