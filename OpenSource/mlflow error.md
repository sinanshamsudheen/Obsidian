tests/webhooks/test_e2e.py::test_prompt_alias_created PASSED | MEM 5.6/16.0 GB | DISK 9.2/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_prompt_webhook_test_endpoint PASSED | MEM 4.7/16.0 GB | DISK 8.8/150.0 GB [100%]

================================== FAILURES ===================================

_________________ test_logging_many_traces_in_single_request __________________

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

method = 'POST', url = '/v1/traces'

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

timeout = Timeout(connect=10, read=10, total=None), chunked = False

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

preload_content = False, decode_content = False, enforce_content_length = True

def _make_request(

self,

conn: BaseHTTPConnection,

method: str,

url: str,

body: _TYPE_BODY | None = None,

headers: typing.Mapping[str, str] | None = None,

retries: Retry | None = None,

timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,

chunked: bool = False,

response_conn: BaseHTTPConnection | None = None,

preload_content: bool = True,

decode_content: bool = True,

enforce_content_length: bool = True,

) -> BaseHTTPResponse:

"""

Perform a request on a given urllib connection object taken from our

pool.

:param conn:

a connection from one of our connection pools

:param method:

HTTP request method (such as GET, POST, PUT, etc.)

:param url:

The URL to perform the request on.

:param body:

Data to send in the request body, either :class:`str`, :class:`bytes`,

an iterable of :class:`str`/:class:`bytes`, or a file-like object.

:param headers:

Dictionary of custom headers to send, such as User-Agent,

If-None-Match, etc. If None, pool headers are used. If provided,

these headers completely replace any pool-specific headers.

:param retries:

Configure the number of retries to allow before raising a

:class:`~urllib3.exceptions.MaxRetryError` exception.

Pass ``None`` to retry until you receive a response. Pass a

:class:`~urllib3.util.retry.Retry` object for fine-grained control

over different types of retries.

Pass an integer number to retry connection errors that many times,

but no other types of errors. Pass zero to never retry.

If ``False``, then retries are disabled and any exception is raised

immediately. Also, instead of raising a MaxRetryError on redirects,

the redirect response will be returned.

:type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

:param timeout:

If specified, overrides the default timeout for this one

request. It may be a float (in seconds) or an instance of

:class:`urllib3.util.Timeout`.

:param chunked:

If True, urllib3 will send the body using chunked transfer

encoding. Otherwise, urllib3 will send the body using the standard

content-length form. Defaults to False.

:param response_conn:

Set this to ``None`` if you will handle releasing the connection or

set the connection to have the response release it.

:param preload_content:

If True, the response's body will be preloaded during construction.

:param decode_content:

If True, will attempt to decode the body based on the

'content-encoding' header.

:param enforce_content_length:

Enforce content length checking. Body returned by server must match

value of Content-Length header, if present. Otherwise, raise error.

"""

self.num_requests += 1

timeout_obj = self._get_timeout(timeout)

timeout_obj.start_connect()

conn.timeout = Timeout.resolve_default_timeout(timeout_obj.connect_timeout)

try:

# Trigger any extra validation we need to do.

try:

self._validate_conn(conn)

except (SocketTimeout, BaseSSLError) as e:

self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)

raise

# _validate_conn() starts the connection to an HTTPS proxy

# so we need to wrap errors with 'ProxyError' here too.

except (

OSError,

NewConnectionError,

TimeoutError,

BaseSSLError,

CertificateError,

SSLError,

) as e:

new_e: Exception = e

if isinstance(e, (BaseSSLError, CertificateError)):

new_e = SSLError(e)

# If the connection didn't successfully connect to it's proxy

# then there

if isinstance(

new_e, (OSError, NewConnectionError, TimeoutError, SSLError)

) and (conn and conn.proxy and not conn.has_connected_to_proxy):

new_e = _wrap_proxy_error(new_e, conn.proxy.scheme)

raise new_e

# conn.request() calls http.client.*.request, not the method in

# urllib3.request. It also calls makefile (recv) on the socket.

try:

conn.request(

method,

url,

body=body,

headers=headers,

chunked=chunked,

preload_content=preload_content,

decode_content=decode_content,

enforce_content_length=enforce_content_length,

)

# We are swallowing BrokenPipeError (errno.EPIPE) since the server is

# legitimately able to close the connection after sending a valid response.

# With this behaviour, the received response is still readable.

except BrokenPipeError:

pass

except OSError as e:

# MacOS/Linux

# EPROTOTYPE and ECONNRESET are needed on macOS

# [https://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/](https://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/)

# Condition changed later to emit ECONNRESET instead of only EPROTOTYPE.

if e.errno != errno.EPROTOTYPE and e.errno != errno.ECONNRESET:

raise

# Reset the timeout for the recv() on the socket

read_timeout = timeout_obj.read_timeout

if not conn.is_closed:

# In Python 3 socket.py will catch EAGAIN and return None when you

# try and read into the file pointer created by http.client, which

# instead raises a BadStatusLine exception. Instead of catching

# the exception and assuming all BadStatusLine exceptions are read

# timeouts, check for a zero timeout before making the request.

if read_timeout == 0:

raise ReadTimeoutError(

self, url, f"Read timed out. (read timeout={read_timeout})"

)

conn.timeout = read_timeout

# Receive the response from the server

try:

> response = conn.getresponse()

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

chunked = False

conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

decode_content = False

enforce_content_length = True

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

method = 'POST'

preload_content = False

read_timeout = 10

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

.venv\lib\site-packages\urllib3\connectionpool.py:534:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

.venv\lib\site-packages\urllib3\connection.py:516: in getresponse

httplib_response = super().getresponse()

HTTPResponse = <class 'urllib3.response.HTTPResponse'>

__class__ = <class 'urllib3.connection.HTTPConnection'>

_shutdown = <built-in method shutdown of socket object at 0x00000241119F9060>

resp_options = _ResponseOptions(request_method='POST', request_url='/v1/traces', preload_content=False, decode_content=False, enforce_content_length=True)

self = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\http\client.py:1375: in getresponse

response.begin()

response = <http.client.HTTPResponse object at 0x0000024111A3A200>

self = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\http\client.py:318: in begin

version, status, reason = self._read_status()

self = <http.client.HTTPResponse object at 0x0000024111A3A200>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\http\client.py:279: in _read_status

line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")

self = <http.client.HTTPResponse object at 0x0000024111A3A200>

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <socket.SocketIO object at 0x0000024111A38A00>

b = <memory at 0x0000024113090040>

def readinto(self, b):

"""Read up to len(b) bytes into the writable buffer *b* and return

the number of bytes read. If the socket is non-blocking and no bytes

are available, None is returned.

If *b* is non-empty, a 0 return value indicates that the connection

was shutdown at the other end.

"""

self._checkClosed()

self._checkReadable()

if self._timeout_occurred:

raise OSError("cannot read from timed out object")

while True:

try:

> return self._sock.recv_into(b)

E TimeoutError: timed out

b = <memory at 0x0000024113090040>

self = <socket.SocketIO object at 0x0000024111A38A00>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\socket.py:705: TimeoutError

The above exception was the direct cause of the following exception:

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

request = <PreparedRequest [POST]>, stream = False

timeout = Timeout(connect=10, read=10, total=None), verify = True, cert = None

proxies = OrderedDict()

def send(

self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None

):

"""Sends PreparedRequest object. Returns Response object.

:param request: The :class:`PreparedRequest <PreparedRequest>` being sent.

:param stream: (optional) Whether to stream the request content.

:param timeout: (optional) How long to wait for the server to send

data before giving up, as a float, or a :ref:`(connect timeout,

read timeout) <timeouts>` tuple.

:type timeout: float or tuple or urllib3 Timeout object

:param verify: (optional) Either a boolean, in which case it controls whether

we verify the server's TLS certificate, or a string, in which case it

must be a path to a CA bundle to use

:param cert: (optional) Any user-provided SSL certificate to be trusted.

:param proxies: (optional) The proxies dictionary to apply to the request.

:rtype: requests.Response

"""

try:

conn = self.get_connection_with_tls_context(

request, verify, proxies=proxies, cert=cert

)

except LocationValueError as e:

raise InvalidURL(e, request=request)

self.cert_verify(conn, request.url, verify, cert)

url = self.request_url(request, proxies)

self.add_headers(

request,

stream=stream,

timeout=timeout,

verify=verify,

cert=cert,

proxies=proxies,

)

chunked = not (request.body is None or "Content-Length" in request.headers)

if isinstance(timeout, tuple):

try:

connect, read = timeout

timeout = TimeoutSauce(connect=connect, read=read)

except ValueError:

raise ValueError(

f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "

f"or a single float to set both timeouts to the same value."

)

elif isinstance(timeout, TimeoutSauce):

pass

else:

timeout = TimeoutSauce(connect=timeout, read=timeout)

try:

> resp = conn.urlopen(

method=request.method,

url=url,

body=request.body,

headers=request.headers,

redirect=False,

assert_same_host=False,

preload_content=False,

decode_content=False,

retries=self.max_retries,

timeout=timeout,

chunked=chunked,

)

cert = None

chunked = False

conn = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

proxies = OrderedDict()

request = <PreparedRequest [POST]>

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

stream = False

timeout = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

verify = True

.venv\lib\site-packages\requests\adapters.py:644:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

.venv\lib\site-packages\urllib3\connectionpool.py:841: in urlopen

retries = retries.increment(

assert_same_host = False

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

body_pos = None

chunked = False

clean_exit = False

conn = None

decode_content = False

destination_scheme = None

err = None

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

http_tunnel_required = False

method = 'POST'

new_e = ReadTimeoutError("HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)")

parsed_url = Url(scheme=None, auth=None, host=None, port=None, path='/v1/traces', query=None, fragment=None)

pool_timeout = None

preload_content = False

redirect = False

release_conn = False

release_this_conn = True

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

response_kw = {}

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

.venv\lib\site-packages\urllib3\util\retry.py:474: in increment

raise reraise(type(error), error, _stacktrace)

_pool = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

_stacktrace = <traceback object at 0x00000241119E3240>

cause = 'unknown'

connect = None

error = ReadTimeoutError("HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)")

method = 'POST'

other = None

read = False

redirect = None

redirect_location = None

response = None

self = Retry(total=0, connect=None, read=False, redirect=None, status=None)

status = None

status_count = None

total = -1

url = '/v1/traces'

.venv\lib\site-packages\urllib3\util\util.py:39: in reraise

raise value

tb = None

tp = <class 'urllib3.exceptions.ReadTimeoutError'>

value = None

.venv\lib\site-packages\urllib3\connectionpool.py:787: in urlopen

response = self._make_request(

assert_same_host = False

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

body_pos = None

chunked = False

clean_exit = False

conn = None

decode_content = False

destination_scheme = None

err = None

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

http_tunnel_required = False

method = 'POST'

new_e = ReadTimeoutError("HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)")

parsed_url = Url(scheme=None, auth=None, host=None, port=None, path='/v1/traces', query=None, fragment=None)

pool_timeout = None

preload_content = False

redirect = False

release_conn = False

release_this_conn = True

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

response_kw = {}

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

.venv\lib\site-packages\urllib3\connectionpool.py:536: in _make_request

self._raise_timeout(err=e, url=url, timeout_value=read_timeout)

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

chunked = False

conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

decode_content = False

enforce_content_length = True

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

method = 'POST'

preload_content = False

read_timeout = 10

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

err = TimeoutError('timed out'), url = '/v1/traces', timeout_value = 10

def _raise_timeout(

self,

err: BaseSSLError | OSError | SocketTimeout,

url: str,

timeout_value: _TYPE_TIMEOUT | None,

) -> None:

"""Is the error actually a timeout? Will raise a ReadTimeout or pass"""

if isinstance(err, SocketTimeout):

> raise ReadTimeoutError(

self, url, f"Read timed out. (read timeout={timeout_value})"

) from err

E urllib3.exceptions.ReadTimeoutError: HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)

err = TimeoutError('timed out')

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout_value = 10

url = '/v1/traces'

.venv\lib\site-packages\urllib3\connectionpool.py:367: ReadTimeoutError

During handling of the above exception, another exception occurred:

mlflow_server = '[http://127.0.0.1:62039](http://127.0.0.1:62039)'

def test_logging_many_traces_in_single_request(mlflow_server: str):

mlflow.set_tracking_uri(mlflow_server)

experiment = mlflow.set_experiment("otel-many-traces-test")

experiment_id = experiment.experiment_id

# Create a request with 15 different traces (exceeds the 10 thread pool limit)

request = ExportTraceServiceRequest()

num_traces = 15

for trace_num in range(num_traces):

span = OTelProtoSpan()

trace_id_hex = f"{trace_num + 1000:016x}" + "0" * 16

span.trace_id = bytes.fromhex(trace_id_hex)

span.span_id = bytes.fromhex(f"{trace_num + 1000:08x}" + "0" * 8)

span.name = f"many-traces-test-span-{trace_num}"

span.start_time_unix_nano = 1000000000 + trace_num * 1000

span.end_time_unix_nano = 2000000000 + trace_num * 1000

scope = InstrumentationScope()

scope.name = "many-traces-test-scope"

scope_spans = ScopeSpans()

scope_spans.scope.CopyFrom(scope)

scope_spans.spans.append(span)

resource = Resource()

resource_spans = ResourceSpans()

resource_spans.resource.CopyFrom(resource)

resource_spans.scope_spans.append(scope_spans)

request.resource_spans.append(resource_spans)

# Send the request and measure response time

> requests.post(

f"{mlflow_server}/v1/traces",

data=request.SerializeToString(),

headers={

"Content-Type": "application/x-protobuf",

MLFLOW_EXPERIMENT_ID_HEADER: experiment_id,

},

timeout=10,

)

experiment = <Experiment: artifact_location='file:///C:/Users/runneradmin/AppData/Local/Temp/pytest-of-runneradmin/pytest-0/test_lo...95, experiment_id='1', last_update_time=1767752261095, lifecycle_stage='active', name='otel-many-traces-test', tags={}>

experiment_id = '1'

mlflow_server = '[http://127.0.0.1:62039](http://127.0.0.1:62039)'

num_traces = 15

request = resource_spans {

resource {

}

scope_spans {

scope {

name: "many-traces-test-scope"

}

spans {

...me: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

}

}

}

resource =

resource_spans = resource {

}

scope_spans {

scope {

name: "many-traces-test-scope"

}

spans {

trace_id: "\000\000\000\000\...00"

name: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

}

}

scope = name: "many-traces-test-scope"

scope_spans = scope {

name: "many-traces-test-scope"

}

spans {

trace_id: "\000\000\000\000\000\000\003\366\000\000\000\000\000\0...\000\000\000"

name: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

}

span = trace_id: "\000\000\000\000\000\000\003\366\000\000\000\000\000\000\000\000"

span_id: "\000\000\003\366\000\000\000\000"

name: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

trace_id_hex = '00000000000003f60000000000000000'

trace_num = 14

tests\tracing\test_otel_logging.py:476:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

.venv\lib\site-packages\requests\api.py:115: in post

return request("post", url, data=data, json=json, **kwargs)

data = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

json = None

kwargs = {'headers': {'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1'}, 'timeout': 10}

url = '[http://127.0.0.1:62039/v1/traces](http://127.0.0.1:62039/v1/traces)'

.venv\lib\site-packages\requests\api.py:59: in request

return session.request(method=method, url=url, **kwargs)

kwargs = {'data': b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00...00', 'headers': {'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1'}, 'json': None, 'timeout': 10}

method = 'post'

session = <requests.sessions.Session object at 0x0000024111A078E0>

url = '[http://127.0.0.1:62039/v1/traces](http://127.0.0.1:62039/v1/traces)'

.venv\lib\site-packages\requests\sessions.py:589: in request

resp = self.send(prep, **send_kwargs)

allow_redirects = True

auth = None

cert = None

cookies = None

data = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

files = None

headers = {'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1'}

hooks = None

json = None

method = 'post'

params = None

prep = <PreparedRequest [POST]>

proxies = {}

req = <Request [POST]>

self = <requests.sessions.Session object at 0x0000024111A078E0>

send_kwargs = {'allow_redirects': True, 'cert': None, 'proxies': OrderedDict(), 'stream': False, ...}

settings = {'cert': None, 'proxies': OrderedDict(), 'stream': False, 'verify': True}

stream = None

timeout = 10

url = '[http://127.0.0.1:62039/v1/traces](http://127.0.0.1:62039/v1/traces)'

verify = None

.venv\lib\site-packages\requests\sessions.py:703: in send

r = adapter.send(request, **kwargs)

adapter = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

allow_redirects = True

hooks = {'response': []}

kwargs = {'cert': None, 'proxies': OrderedDict(), 'stream': False, 'timeout': 10, ...}

request = <PreparedRequest [POST]>

self = <requests.sessions.Session object at 0x0000024111A078E0>

start = 1666.4083634

stream = False

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

request = <PreparedRequest [POST]>, stream = False

timeout = Timeout(connect=10, read=10, total=None), verify = True, cert = None

proxies = OrderedDict()

def send(

self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None

):

"""Sends PreparedRequest object. Returns Response object.

:param request: The :class:`PreparedRequest <PreparedRequest>` being sent.

:param stream: (optional) Whether to stream the request content.

:param timeout: (optional) How long to wait for the server to send

data before giving up, as a float, or a :ref:`(connect timeout,

read timeout) <timeouts>` tuple.

:type timeout: float or tuple or urllib3 Timeout object

:param verify: (optional) Either a boolean, in which case it controls whether

we verify the server's TLS certificate, or a string, in which case it

must be a path to a CA bundle to use

:param cert: (optional) Any user-provided SSL certificate to be trusted.

:param proxies: (optional) The proxies dictionary to apply to the request.

:rtype: requests.Response

"""

try:

conn = self.get_connection_with_tls_context(

request, verify, proxies=proxies, cert=cert

)

except LocationValueError as e:

raise InvalidURL(e, request=request)

self.cert_verify(conn, request.url, verify, cert)

url = self.request_url(request, proxies)

self.add_headers(

request,

stream=stream,

timeout=timeout,

verify=verify,

cert=cert,

proxies=proxies,

)

chunked = not (request.body is None or "Content-Length" in request.headers)

if isinstance(timeout, tuple):

try:

connect, read = timeout

timeout = TimeoutSauce(connect=connect, read=read)

except ValueError:

raise ValueError(

f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "

f"or a single float to set both timeouts to the same value."

)

elif isinstance(timeout, TimeoutSauce):

pass

else:

timeout = TimeoutSauce(connect=timeout, read=timeout)

try:

resp = conn.urlopen(

method=request.method,

url=url,

body=request.body,

headers=request.headers,

redirect=False,

assert_same_host=False,

preload_content=False,

decode_content=False,

retries=self.max_retries,

timeout=timeout,

chunked=chunked,

)

except (ProtocolError, OSError) as err:

raise ConnectionError(err, request=request)

except MaxRetryError as e:

if isinstance(e.reason, ConnectTimeoutError):

# TODO: Remove this in 3.0.0: see #2811

if not isinstance(e.reason, NewConnectionError):

raise ConnectTimeout(e, request=request)

if isinstance(e.reason, ResponseError):

raise RetryError(e, request=request)

if isinstance(e.reason, _ProxyError):

raise ProxyError(e, request=request)

if isinstance(e.reason, _SSLError):

# This branch is for urllib3 v1.22 and later.

raise SSLError(e, request=request)

raise ConnectionError(e, request=request)

except ClosedPoolError as e:

raise ConnectionError(e, request=request)

except _ProxyError as e:

raise ProxyError(e)

except (_SSLError, _HTTPError) as e:

if isinstance(e, _SSLError):

# This branch is for urllib3 versions earlier than v1.22

raise SSLError(e, request=request)

elif isinstance(e, ReadTimeoutError):

> raise ReadTimeout(e, request=request)

E requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)

cert = None

chunked = False

conn = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

proxies = OrderedDict()

request = <PreparedRequest [POST]>

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

stream = False

timeout = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

verify = True

.venv\lib\site-packages\requests\adapters.py:690: ReadTimeout

---------------------------- Captured stderr setup ----------------------------

2026/01/07 02:17:40 INFO mlflow.store.db.utils: Create SQLAlchemy engine with pool options {'poolclass': <class 'sqlalchemy.pool.impl.NullPool'>}

2026/01/07 02:17:40 INFO alembic.runtime.migration: Context impl SQLiteImpl.

2026/01/07 02:17:40 INFO alembic.runtime.migration: Will assume non-transactional DDL.

2026/01/07 02:17:40 INFO mlflow.store.db.utils: Create SQLAlchemy engine with pool options {'poolclass': <class 'sqlalchemy.pool.impl.NullPool'>}

---------------------------- Captured stderr call -----------------------------

2026/01/07 02:17:41 INFO mlflow.tracking.fluent: Experiment with name 'otel-many-traces-test' does not exist. Creating a new experiment.

-------------------------- Captured stderr teardown ---------------------------

2026/01/07 02:17:58 INFO mlflow.tracking.fluent: Active model is cleared

============================== warnings summary ===============================

mlflow\pyfunc\utils\data_validation.py:186

mlflow\pyfunc\utils\data_validation.py:186

mlflow\pyfunc\utils\data_validation.py:186

tests/store/model_registry/test_file_store.py::test_create_model_version_with_model_id_and_no_run_id

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

tests/tracking/fluent/test_fluent.py::test_last_logged_model_log_model

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:186: UserWarning: Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check [https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel](https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel) for more details.

color_warning(

tests\metrics\test_metric_definitions.py:38

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:38: FutureWarning: ``mlflow.metrics.ari_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

ari_grade_level(),

tests\metrics\test_metric_definitions.py:39

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:39: FutureWarning: ``mlflow.metrics.exact_match`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

exact_match(),

tests\metrics\test_metric_definitions.py:40

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:40: FutureWarning: ``mlflow.metrics.flesch_kincaid_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

flesch_kincaid_grade_level(),

tests\metrics\test_metric_definitions.py:41

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:41: FutureWarning: ``mlflow.metrics.rouge1`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rouge1(),

tests\metrics\test_metric_definitions.py:42

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:42: FutureWarning: ``mlflow.metrics.rouge2`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rouge2(),

tests\metrics\test_metric_definitions.py:43

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:43: FutureWarning: ``mlflow.metrics.rougeL`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rougeL(),

tests\metrics\test_metric_definitions.py:44

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:44: FutureWarning: ``mlflow.metrics.rougeLsum`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rougeLsum(),

tests\metrics\test_metric_definitions.py:45

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:45: FutureWarning: ``mlflow.metrics.toxicity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

toxicity(),

tests\metrics\test_metric_definitions.py:46

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:46: FutureWarning: ``mlflow.metrics.bleu`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

bleu(),

.venv\lib\site-packages\google\api_core\_python_version_support.py:266

D:\a\mlflow\mlflow\.venv\lib\site-packages\google\api_core\_python_version_support.py:266: FutureWarning: You are using a Python version (3.10.11) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.

warnings.warn(message, FutureWarning)

.venv\lib\site-packages\starlette\middleware\wsgi.py:14

D:\a\mlflow\mlflow\.venv\lib\site-packages\starlette\middleware\wsgi.py:14: DeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to [https://github.com/abersheeran/a2wsgi](https://github.com/abersheeran/a2wsgi) as a replacement.

warnings.warn(

tests/data/test_artifact_dataset_sources.py::test_expected_artifact_dataset_sources_are_registered_and_resolvable[file:///tmp/path/to/my/local/directory-local-LocalArtifactDatasetSource]

tests/data/test_artifact_dataset_sources.py::test_to_and_from_json[file:///tmp/path/to/my/local/directory-local]

tests/data/test_artifact_dataset_sources.py::test_load_makes_expected_mlflow_artifacts_download_call[file:///tmp/path/to/my/local/directory-local]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'file:///tmp/path/to/my/local/directory'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py: 4 warnings

tests/data/test_meta_dataset.py: 1 warning

tests/data/test_pandas_dataset.py: 1 warning

tests/data/test_polars_dataset.py: 1 warning

tests/data/test_tensorflow_dataset.py: 2 warnings

tests/tracking/fluent/test_fluent.py: 1 warning

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: The specified dataset source can be interpreted in multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource. MLflow will assume that this is a LocalArtifactDatasetSource source.

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_expected_artifact_dataset_sources_are_registered_and_resolvable[wasbs://user@host.blob.core.windows.net/dir-wasbs-AzureBlobArtifactDatasetSource]

tests/data/test_artifact_dataset_sources.py::test_to_and_from_json[wasbs://user@host.blob.core.windows.net/dir-wasbs]

tests/data/test_artifact_dataset_sources.py::test_load_makes_expected_mlflow_artifacts_download_call[wasbs://user@host.blob.core.windows.net/dir-wasbs]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'wasbs://user@host.blob.core.windows.net/dir'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_expected_artifact_dataset_sources_are_registered_and_resolvable[hdfs://host_name:8020/hdfs/path/to/my/file.txt-hdfs-HdfsArtifactDatasetSource]

tests/data/test_artifact_dataset_sources.py::test_to_and_from_json[hdfs://host_name:8020/hdfs/path/to/my/file.txt-hdfs]

tests/data/test_artifact_dataset_sources.py::test_load_makes_expected_mlflow_artifacts_download_call[hdfs://host_name:8020/hdfs/path/to/my/file.txt-hdfs]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'hdfs://host_name:8020/hdfs/path/to/my/file.txt'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_local_load[dst]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_local_load_dst_0\myfile.txt'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_local_load[dst]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_local_load_dst_0\mydir'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_dataset_source.py::test_get_source_obtains_expected_code_source

tests/data/test_huggingface_dataset_and_source.py::test_dataset_conversion_to_json

tests/data/test_meta_dataset.py::test_create_meta_dataset_from_dataset

tests/data/test_pandas_dataset.py::test_from_pandas_file_system_datasource

tests/telemetry/test_tracked_events.py::test_evaluate

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

D:\a\mlflow\mlflow\mlflow\types\utils.py:452: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <[https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values](https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values)>`_ for more details.

warnings.warn(

tests/data/test_dataset_source_registry.py::test_resolve_dataset_source_maintains_consistent_order_and_uses_last_registered_match

D:\a\mlflow\mlflow\tests\data\test_dataset_source_registry.py:108: UserWarning: The specified dataset source can be interpreted in multiple ways: SampleDatasetSource, SampleDatasetSourceCopy1, SampleDatasetSourceCopy2. MLflow will assume that this is a SampleDatasetSourceCopy2 source.

source1 = registry1.resolve("test:/" + str(tmp_path))

tests/data/test_dataset_source_registry.py::test_resolve_dataset_source_maintains_consistent_order_and_uses_last_registered_match

D:\a\mlflow\mlflow\tests\data\test_dataset_source_registry.py:115: UserWarning: The specified dataset source can be interpreted in multiple ways: SampleDatasetSource, SampleDatasetSourceCopy2, SampleDatasetSourceCopy1. MLflow will assume that this is a SampleDatasetSourceCopy1 source.

source2 = registry2.resolve("test:/" + str(tmp_path))

tests/data/test_dataset_source_registry.py::test_resolve_dataset_source_maintains_consistent_order_and_uses_last_registered_match

D:\a\mlflow\mlflow\tests\data\test_dataset_source_registry.py:133: UserWarning: The specified dataset source can be interpreted in multiple ways: DummyDatasetSourceCopy, DummyDatasetSource. MLflow will assume that this is a DummyDatasetSource source.

source5 = registry3.resolve("dummy:/" + str(tmp_path))

tests/data/test_meta_dataset.py::test_create_meta_dataset_from_dataset

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for '/tmp/test.csv'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_pandas_dataset.py::test_from_pandas_file_system_datasource

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_from_pandas_file_system_d0\temp.csv'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_polars_dataset.py::test_from_polars_with_targets

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_from_polars_with_targets0\temp.csv'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_tensorflow_dataset.py::test_from_tensorflow_dataset_constructs_expected_dataset

tests/data/test_tensorflow_dataset.py::test_from_tensorflow_tensor_with_targets_constructs_expected_dataset

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'my_source'. Exception:

return _dataset_source_registry.resolve(

tests/deployments/databricks/test_databricks.py::test_update_endpoint

D:\a\mlflow\mlflow\tests\deployments\databricks\test_databricks.py:366: FutureWarning: ``mlflow.deployments.databricks.DatabricksDeploymentClient.update_endpoint`` is deprecated. This method will be removed in a future release. Use ``update_endpoint_config, update_endpoint_tags, update_endpoint_rate_limits, or update_endpoint_ai_gateway`` instead.

resp = client.update_endpoint(

tests/entities/test_assessment_source.py::test_assessment_source_type_validation

D:\a\mlflow\mlflow\mlflow\entities\assessment_source.py:169: FutureWarning: AI_JUDGE is deprecated. Use LLM_JUDGE instead.

warnings.warn(

tests/entities/test_trace_data.py::test_intermediate_outputs_from_attribute

D:\a\mlflow\mlflow\tests\entities\test_trace_data.py:168: FutureWarning: ``mlflow.entities.trace_data.TraceData.intermediate_outputs`` is deprecated since 3.6.0. This method will be removed in a future release. Use ``trace.search_spans(name=...)`` instead.

assert trace.data.intermediate_outputs == intermediate_outputs

tests/entities/test_trace_location.py::test_trace_location_mismatch

D:\a\mlflow\mlflow\tests\entities\test_trace_location.py:58: FutureWarning: ``mlflow.entities.trace_location.InferenceTableLocation`` is deprecated since 3.7.0. This method will be removed in a future release.

inference_table=InferenceTableLocation(full_table_name="a.b.c"),

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_correct_response

tests/metrics/genai/test_genai_metrics.py::test_log_make_genai_metric_fn_args

tests/metrics/genai/test_genai_metrics.py::test_genai_metrics_callable

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:148: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_correct_response

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:188: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

custom_metric = make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_malformed_input_raises_exception

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:397: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_similarity_metric = answer_similarity()

tests/metrics/genai/test_genai_metrics.py::test_malformed_input_raises_exception

tests/metrics/genai/test_genai_metrics.py::test_similarity_metric[parameters1-None-None]

tests/metrics/genai/test_genai_metrics.py::test_genai_metric_with_custom_chat_endpoint[True]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:85: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_throws_if_grading_context_cols_wrong[good_column-bad_column]

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_throws_if_grading_context_cols_wrong[grading_cols3-example_context_cols3]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:547: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_similarity_metric[parameters1-None-None]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:647: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

similarity_metric = answer_similarity(

tests/metrics/genai/test_genai_metrics.py::test_similarity_metric[parameters1-None-None]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:719: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_similarity(

tests/metrics/genai/test_genai_metrics.py::test_answer_correctness_metric

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:806: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_correctness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_correctness_metric = answer_correctness()

tests/metrics/genai/test_genai_metrics.py::test_answer_correctness_metric

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[answer_correctness]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:178: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_answer_correctness_metric

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:871: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_correctness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_correctness(metric_version="non-existent-version")

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_metric_details

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1023: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

custom_metric = make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_make_custom_judge_prompt_genai_metric

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1089: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric_from_prompt`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

custom_judge_prompt_metric = make_genai_metric_from_prompt(

tests/metrics/genai/test_genai_metrics.py::test_metric_metadata_on_prebuilt_genai_metrics[faithfulness]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1234: FutureWarning: ``mlflow.metrics.genai.metric_definitions.faithfulness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric = metric_fn(metric_metadata={"metadata_field": "metadata_value"})

tests/metrics/genai/test_genai_metrics.py::test_metric_metadata_on_prebuilt_genai_metrics[faithfulness]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:266: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_genai_metric_with_custom_chat_endpoint[True]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1335: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

similarity_metric = answer_similarity(

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[answer_correctness]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1392: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_correctness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric_fn(parameters={"temperature": 0.1})

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[relevance]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1392: FutureWarning: ``mlflow.metrics.genai.metric_definitions.relevance`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric_fn(parameters={"temperature": 0.1})

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[relevance]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:441: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/test_metric_definitions.py::test_flesch_kincaid_grade_level

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:90: FutureWarning: ``mlflow.metrics.flesch_kincaid_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = flesch_kincaid_grade_level().eval_fn(predictions, None, {})

tests/metrics/test_metric_definitions.py::test_flesch_kincaid_grade_level

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:95: FutureWarning: ``mlflow.metrics.flesch_kincaid_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert flesch_kincaid_grade_level()(predictions=predictions) == result

tests/metrics/test_metric_definitions.py::test_rouge1

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:133: FutureWarning: ``mlflow.metrics.rouge1`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = rouge1().eval_fn(predictions, targets, {})

tests/metrics/test_metric_definitions.py::test_rouge1

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:139: FutureWarning: ``mlflow.metrics.rouge1`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert rouge1()(predictions=predictions, targets=targets) == result

tests/metrics/test_metric_definitions.py::test_rougeLsum

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:169: FutureWarning: ``mlflow.metrics.rougeLsum`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = rougeLsum().eval_fn(predictions, targets, {})

tests/metrics/test_metric_definitions.py::test_rougeLsum

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:175: FutureWarning: ``mlflow.metrics.rougeLsum`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert rougeLsum()(predictions=predictions, targets=targets) == result

tests/metrics/test_metric_definitions.py::test_recall_at_k

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:287: FutureWarning: ``mlflow.metrics.recall_at_k`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = recall_at_k(4).eval_fn(predictions, targets)

tests/metrics/test_metric_definitions.py::test_recall_at_k

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:295: FutureWarning: ``mlflow.metrics.recall_at_k`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert recall_at_k(4)(predictions=predictions, targets=targets) == result

tests/metrics/test_metric_definitions.py::test_builtin_metric_call_signature

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:391: FutureWarning: ``mlflow.metrics.ndcg_at_k`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric = ndcg_at_k(3)

tests/store/model_registry/test_rest_store_webhooks.py::test_create_webhook

D:\a\mlflow\mlflow\.venv\lib\site-packages\websockets\legacy\__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see [https://websockets.readthedocs.io/en/stable/howto/upgrade.html](https://websockets.readthedocs.io/en/stable/howto/upgrade.html) for upgrade instructions

warnings.warn( # deprecated in 14.0 - 2024-11-09

tests/store/model_registry/test_rest_store_webhooks.py::test_create_webhook

D:\a\mlflow\mlflow\.venv\lib\site-packages\uvicorn\protocols\websockets\websockets_impl.py:17: DeprecationWarning: websockets.server.WebSocketServerProtocol is deprecated

from websockets.server import WebSocketServerProtocol

tests/store/tracking/test_sqlalchemy_store.py::test_run_needs_uuid

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:187: SAWarning: Column 'runs.run_uuid' is marked as a member of the primary key for table 'runs', but has no Python-side or server-side default generator indicated, nor does it indicate 'autoincrement=True' or 'nullable=True', and no explicit value is passed. Primary key columns typically may not store NULL.

session.commit()

tests/store/tracking/test_sqlalchemy_store.py::test_delete_traces_with_max_count

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

tests/tracing/test_provider.py::test_disable_enable_tracing

tests/tracing/utils/test_copy.py::test_copy_trace_missing_info

D:\a\mlflow\mlflow\mlflow\store\tracking\sqlalchemy_store.py:3002: SAWarning: Coercing Subquery object into a select() for use in IN(); please pass a select() construct explicitly

SqlTraceInfo.request_id.in_(

tests/telemetry/test_tracked_events.py::test_evaluate

D:\a\mlflow\mlflow\tests\telemetry\test_tracked_events.py:349: FutureWarning: ``mlflow.metrics.latency`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

extra_metrics=[mlflow.metrics.latency()],

tests/telemetry/test_tracked_events.py::test_invoke_custom_judge_model[endpoints:/my-endpoint-endpoints-True-False]

D:\a\mlflow\mlflow\tests\telemetry\test_tracked_events.py:992: FutureWarning: The legacy provider 'endpoints' is deprecated and will be removed in a future release. Please update your code to use the 'databricks' provider instead.

invoke_judge_model(

tests/tracing/test_fluent.py::test_trace_with_experiment_id_issue_warning_when_not_root_span

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:652: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

@mlflow.trace(trace_destination=MlflowExperiment(exp_1))

tests/tracing/test_fluent.py::test_trace_with_experiment_id_issue_warning_when_not_root_span

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:656: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

@mlflow.trace(trace_destination=MlflowExperiment(exp_1))

tests/tracing/test_fluent.py::test_search_traces[list]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:913: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_fluent.py::test_search_traces_with_pagination

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:974: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(experiment_ids=["1"])

tests/tracing/test_fluent.py::test_search_traces_with_non_dict_span_inputs_outputs

tests/tracing/test_fluent.py::test_search_traces_with_non_existent_field

tests/tracing/test_fluent.py::test_search_traces_invalid_extract_fields[extract_fields0]

D:\a\mlflow\mlflow\mlflow\utils\annotations.py:260: FutureWarning: The `extract_fields` parameter is deprecated and will be removed in a future version.

return func(*args, **kwargs)

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:155: FutureWarning: Model's `predict` method contains invalid parameters: {'model_inputs'}. Only the following parameter names are allowed: context, model_input, and params. Note that invalid parameters will no longer be permitted in future versions.

param_names = _check_func_signature(func, "predict")

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-outputs0-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-Yes-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-None-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[None-outputs0-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[None-Yes-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[None-None-inputs0]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2134: FutureWarning: ``mlflow.tracing.fluent.log_trace`` is deprecated since 3.6.0. This method will be removed in a future release.

mlflow.log_trace(

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-outputs0-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-Yes-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-None-inputs0]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2153: FutureWarning: ``mlflow.entities.trace_data.TraceData.intermediate_outputs`` is deprecated since 3.6.0. This method will be removed in a future release. Use ``trace.search_spans(name=...)`` instead.

assert trace.data.intermediate_outputs == intermediate_outputs

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2175: FutureWarning: Parameter 'request_id' is deprecated. Please use 'trace_id' instead.

mlflow.set_trace_tag(request_id=trace_id, key="key3", value="value3")

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2176: FutureWarning: Parameter 'request_id' is deprecated. Please use 'trace_id' instead.

trace = mlflow.get_trace(request_id=trace_id)

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2179: FutureWarning: Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

mlflow.delete_trace_tag(request_id=trace_id, key="key3")

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2180: FutureWarning: Parameter 'request_id' is deprecated. Please use 'trace_id' instead.

trace = mlflow.get_trace(request_id=trace_id)

tests/tracing/test_fluent.py::test_set_destination_in_threads[True]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2279: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

set_destination(MlflowExperiment(experiment_id1))

tests/tracing/test_fluent.py::test_set_destination_in_threads[True]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2272: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

set_destination(MlflowExperiment(experiment_id), context_local=True)

tests/tracing/test_fluent.py::test_set_destination_in_async_contexts[False]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2340: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

set_destination(MlflowExperiment(experiment_id), context_local=True)

tests/tracing/test_otel_loading.py::test_get_trace_for_otel_sent_span[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:110: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_get_trace_with_otel_span_events[True]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:198: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_get_trace_with_otel_span_status[True]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:238: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_set_trace_tag_on_otel_trace[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:265: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_log_feedback_on_otel_trace[True]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:331: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_multiple_assessments_on_otel_trace[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:390: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_multiple_assessments_on_otel_trace[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:445: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tagged_traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_span_inputs_outputs_translation[True-GenAiTranslator]

tests/tracing/test_otel_loading.py::test_span_inputs_outputs_translation[False-GenAiTranslator]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:508: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_span_token_usage_translation[True-GenAiTranslator]

tests/tracing/test_otel_loading.py::test_span_token_usage_translation[False-GenAiTranslator]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:539: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_aggregated_token_usage_from_multiple_spans[True-GenAiTranslator]

tests/tracing/test_otel_loading.py::test_aggregated_token_usage_from_multiple_spans[False-GenAiTranslator]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:582: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_logging.py::test_otel_client_sends_spans_to_mlflow_database

D:\a\mlflow\mlflow\tests\tracing\test_otel_logging.py:127: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_tracing_client.py::test_tracing_client_link_prompt_versions_to_trace

D:\a\mlflow\mlflow\tests\tracing\test_tracing_client.py:52: FutureWarning: The `mlflow.register_prompt` API is moved to the `mlflow.genai` namespace. Please use `mlflow.genai.register_prompt` instead. The original API will be removed in the future release.

prompt_version = mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:183: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

session.execute(sql.text("PRAGMA foreign_keys = ON;"))

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:184: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

session.execute(sql.text("PRAGMA busy_timeout = 20000;"))

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:185: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

session.execute(sql.text("PRAGMA case_sensitive_like = true;"))

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store_with_artifact_uri[mysql]

D:\a\mlflow\mlflow\mlflow\store\tracking\sqlalchemy_store.py:486: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

.one_or_none()

tests/tracking/fluent/test_fluent.py::test_log_input_polars

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_log_input_polars0\temp.csv'. Exception:

return _dataset_source_registry.resolve(

tests/tracking/test_client.py::test_start_and_end_trace_does_not_log_trace_when_disabled[file-True]

tests/tracking/test_client.py::test_start_and_end_trace_does_not_log_trace_when_disabled[sqlalchemy-False]

D:\a\mlflow\mlflow\tests\tracking\test_client.py:1028: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

assert client.search_traces(experiment_ids=[experiment_id]) == []

tests/tracking/test_client.py::test_transition_model_version_stage

D:\a\mlflow\mlflow\tests\tracking\test_client.py:1449: FutureWarning: ``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

actual_result = MlflowClient(registry_uri="sqlite:///somedb.db").transition_model_version_stage(

tests/tracking/test_client.py::test_crud_prompts[file]

D:\a\mlflow\mlflow\mlflow\prompt\registry_utils.py:209: FutureWarning: The `mlflow.load_prompt` API is moved to the `mlflow.genai` namespace. Please use `mlflow.genai.load_prompt` instead. The original API will be removed in the future release.

return func(*args, **kwargs)

tests/tracking/test_client.py::test_block_handling_prompt_with_model_apis[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_client.py:2363: FutureWarning: ``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

api(*args)

tests/tracking/test_client.py::test_block_handling_prompt_with_model_apis[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_client.py:2363: FutureWarning: ``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

api(*args)

tests/tracking/test_model_registry.py::test_search_registered_model_flow_paginated[file-None-<lambda>-100]

D:\a\mlflow\mlflow\.venv\lib\site-packages\_pytest\unraisableexception.py:65: PytestUnraisableExceptionWarning:

Exception ignored in: <function Variable.__del__ at 0x0000024111DE3760>

Traceback (most recent call last):

File "C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\tkinter\__init__.py", line 388, in __del__

if self._tk.getboolean(self._tk.call("info", "exists", self._name)):

RuntimeError: main thread is not in main loop

Enable tracemalloc to get traceback where the object was allocated.

See [https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings](https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings) for more info.

tests/tracking/test_model_registry.py::test_search_registered_model_flow_paginated[file-None-<lambda>-100]

D:\a\mlflow\mlflow\.venv\lib\site-packages\_pytest\unraisableexception.py:65: PytestUnraisableExceptionWarning:

Exception ignored in: <function Image.__del__ at 0x000002411A89F400>

Traceback (most recent call last):

File "C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\tkinter\__init__.py", line 4056, in __del__

self.tk.call('image', 'delete', self.name)

RuntimeError: main thread is not in main loop

Enable tracemalloc to get traceback where the object was allocated.

See [https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings](https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings) for more info.

tests/tracking/test_model_registry.py::test_update_model_version_flow[file]

D:\a\mlflow\mlflow\tests\tracking\test_model_registry.py:486: FutureWarning:

``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

tests/tracking/test_model_registry.py::test_latest_models[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_model_registry.py:539: FutureWarning:

``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

tests/tracking/test_model_registry.py::test_latest_models[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_model_registry.py:544: FutureWarning:

``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

tests/tracking/test_rest_tracking.py::test_log_input[file]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning:

Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_log_input_file_0\temp.csv'. Exception:

tests/tracking/test_rest_tracking.py::test_log_input[file]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning:

The specified dataset source can be interpreted in multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource. MLflow will assume that this is a LocalArtifactDatasetSource source.

tests/tracking/test_rest_tracking.py::test_log_input[file]

tests/types/test_schema.py::test_schema_inference_on_dictionary_of_strings[data3-schema3]

tests/types/test_schema.py::test_spark_schema_inference

tests/types/test_schema.py::test_spark_schema_inference

tests/types/test_schema.py::test_schema_inference_on_datatypes[data5-DataType.long]

tests/types/test_type_hints.py::test_convert_dataframe_to_example_format[data5]

tests/types/test_type_hints.py::test_convert_dataframe_to_example_format[data8]

tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

tests/utils/test_requirements_utils.py::test_gateway_extra_not_captured_when_importing_deployment_client_only

D:\a\mlflow\mlflow\mlflow\types\utils.py:452: UserWarning:

Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <[https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values](https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values)>`_ for more details.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

tests/tracking/test_rest_tracking.py::test_search_traces[file]

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2648: FutureWarning:

Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2659: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2663: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2671: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2677: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2738: FutureWarning:

Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

tests/tracking/test_rest_tracking.py: 10 warnings

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2795: FutureWarning:

Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

tests/tracking/test_rest_tracking.py::test_link_traces_to_run_and_search_traces[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2993: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_link_traces_to_run_and_search_traces[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2997: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_get_logged_model_artifact[file]

tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

tests/utils/test_requirements_utils.py::test_gateway_extra_not_captured_when_importing_deployment_client_only

tests/utils/test_requirements_utils.py::test_capture_imported_modules_with_exception

tests/utils/test_requirements_utils.py::test_capture_imported_modules_extra_env_vars

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:186: UserWarning:

Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check [https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel](https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel) for more details.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\tests\types\test_schema.py:1784: FutureWarning:

``mlflow.models.rag_signatures.ChatCompletionRequest`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatCompletionRequest`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\mlflow\models\rag_signatures.py:25: FutureWarning:

``mlflow.models.rag_signatures.Message`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatMessage`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\tests\types\test_schema.py:1801: FutureWarning:

``mlflow.models.rag_signatures.ChatCompletionResponse`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatCompletionResponse`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\mlflow\models\rag_signatures.py:71: FutureWarning:

``mlflow.models.rag_signatures.ChainCompletionChoice`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatChoice`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\mlflow\models\rag_signatures.py:47: FutureWarning:

``mlflow.models.rag_signatures.Message`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatMessage`` instead.

tests/types/test_type_hints.py::test_infer_schema_from_pydantic_model[list-expected_schema0]

D:\a\mlflow\mlflow\mlflow\types\type_hints.py:287: UserWarning:

Any type hint is inferred as AnyType, and MLflow doesn't validate the data for this type. Please use a more specific type hint to enable data validation.

tests/types/test_type_hints.py::test_infer_schema_from_python_type_hints[list-expected_schema18]

D:\a\mlflow\mlflow\mlflow\types\type_hints.py:385: UserWarning:

Any type hint is inferred as AnyType, and MLflow doesn't validate the data for this type. Please use a more specific type hint to enable data validation.

tests/utils/test_annotations.py::test_deprecated_dataclass_preserves_fields

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:270: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_annotations.py::test_deprecated_dataclass_dunder_methods_not_mutated

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:288: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_annotations.py::test_deprecated_dataclass_dunder_methods_not_mutated

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:295: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_annotations.py::test_deprecated_dataclass_dunder_methods_not_mutated

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:296: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:155: FutureWarning:

Model's `predict` method contains invalid parameters: {'inputs'}. Only the following parameter names are allowed: context, model_input, and params. Note that invalid parameters will no longer be permitted in future versions.

-- Docs: [https://docs.pytest.org/en/stable/how-to/capture-warnings.html](https://docs.pytest.org/en/stable/how-to/capture-warnings.html)

============================ slowest 10 durations =============================

30.18s setup tests/webhooks/test_e2e.py::test_model_version_created

30.02s call tests/tracking/test_client.py::test_client_get_trace_empty_result

24.67s setup tests/server/test_prometheus_exporter.py::test_metrics

24.33s setup tests/data/test_delta_dataset_source.py::test_delta_dataset_source_from_path

24.25s setup tests/tracking/test_client_webhooks.py::test_get_webhook_not_found

23.89s call tests/tracking/fluent/test_fluent.py::test_search_experiments

23.52s call tests/tracking/test_model_registry.py::test_search_registered_model_flow_paginated[file-None-<lambda>-100]

21.88s setup tests/tracking/test_rest_tracking.py::test_update_secret

20.48s call tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

19.99s call tests/tracking/_tracking_service/test_utils.py::test_get_store_with_empty_mlruns

========================= per-file durations (sorted) =========================

per-file durations (sorted)

========================= command to run failed tests =========================

pytest 'tests/tracing/test_otel_logging.py::test_logging_many_traces_in_single_request'

============================== Remaining threads ==============================

1: <TMonitor(Thread-1, started daemon 6376)>

2: <TMonitor(Thread-2, started daemon 8476)>

3: <TMonitor(Thread-14, started daemon 3164)>

4: <Thread(MlflowAutologgingQueueingClient_0, started 5936)>

5: <FinalizerWorker(Thread-99, started daemon 4020)>

6: <Thread(EphemeralCacheEncryption-cleanup, started daemon 8424)>

7: <Thread(EphemeralCacheEncryption-cleanup, started daemon 2640)>

8: <Thread(EphemeralCacheEncryption-cleanup, started daemon 9412)>

9: <Thread(EphemeralCacheEncryption-cleanup, started daemon 6456)>

10: <Thread(EphemeralCacheEncryption-cleanup, started daemon 9980)>

11: <Thread(EphemeralCacheEncryption-cleanup, started daemon 6692)>

12: <Thread(EphemeralCacheEncryption-cleanup, started daemon 6668)>

13: <Thread(EphemeralCacheEncryption-cleanup, started daemon 7612)>

14: <Thread(MLflowSpanBatcherWorker, started daemon 9048)>

15: <Thread(MLflowSpanBatcherWorker, started daemon 8552)>

16: <Thread(MLflowTraceLoggingConsumer, started daemon 772)>

17: <Thread(MlflowTraceLoggingWorker_0, started daemon 8316)>

18: <Thread(OtelPeriodicExportingMetricReader, started daemon 4840)>

19: <Thread(MLflowTraceLoggingConsumer, started daemon 1252)>

20: <Thread(MlflowTraceLoggingWorker_0, started daemon 8204)>

21: <Thread(MlflowTraceLoggingWorker_1, started daemon 9844)>

22: <Thread(MlflowTraceLoggingWorker_2, started daemon 4848)>

23: <Thread(MLflowSpanBatcherWorker, started daemon 6080)>

24: <Thread(MLflowSpanBatcherWorker, started daemon 3224)>

25: <Thread(MLflowSpanBatcherWorker, started daemon 8284)>

26: <Thread(OtelBatchSpanRecordProcessor, started daemon 10196)>

27: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 8808)>

28: <Thread(MlflowLocalArtifactRepository_0, started 9224)>

29: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 5004)>

30: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 7796)>

31: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_0, started 9396)>

32: <Thread(MLflowArtifactsLoggingWorkerPool_0, started daemon 2100)>

33: <Thread(MLflowArtifactsLoggingWorkerPool_1, started daemon 1636)>

34: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_1, started 7576)>

35: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_2, started 1564)>

36: <Thread(MLflowArtifactsLoggingWorkerPool_2, started daemon 840)>

37: <Thread(MLflowArtifactsLoggingWorkerPool_3, started daemon 7300)>

38: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_3, started 5648)>

39: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_4, started 7784)>

40: <Thread(MLflowArtifactsLoggingWorkerPool_4, started daemon 7864)>

41: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 8936)>

42: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_0, started 6716)>

43: <Thread(MLflowArtifactsLoggingWorkerPool_0, started daemon 8116)>

44: <Thread(MLflowArtifactsLoggingWorkerPool_1, started daemon 1396)>

45: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_1, started 7340)>

46: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_2, started 1732)>

47: <Thread(MLflowArtifactsLoggingWorkerPool_2, started daemon 9384)>

48: <Thread(MLflowArtifactsLoggingWorkerPool_3, started daemon 8668)>

49: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_3, started 4636)>

50: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_4, started 4524)>

51: <Thread(MLflowArtifactsLoggingWorkerPool_4, started daemon 5064)>

52: <Thread(Thread-1999 (serve_forever), started daemon 8640)>

========================== Remaining child processes ==========================

1: psutil.Process(pid=8500, name='cmd.exe', status='running', started='02:05:46')

2: psutil.Process(pid=9056, name='cmd.exe', status='running', started='02:05:46')

3: psutil.Process(pid=8632, name='java.exe', status='running', started='02:05:53')

=========================== short test summary info ===========================

FAILED | MEM 4.7/16.0 GB | DISK 8.8/150.0 GB tests/tracing/test_otel_logging.py::test_logging_many_traces_in_single_request - requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)

= 1 failed, 2204 passed, 59 skipped, 1 xfailed, 207 warnings in 1462.26s (0:24:22) =

2026/01/07 02:28:38 INFO mlflow.tracing.export.async_export_queue: Flushing the async trace logging queue before program exit. This may take a while...

2026/01/07 02:28:39 INFO mlflow.tracing.export.async_export_queue: Flushing the async trace logging queue before program exit. This may take a while...

SUCCESS: The process with PID 8632 (child process of PID 9056) has been terminated.

SUCCESS: The process with PID 9056 (child process of PID 8500) has been terminated.

SUCCESS: The process with PID 8500 (child process of PID 1376) has been terminated.

24m 47s

tests/utils/test_validation.py::test_validate_experiment_name_bad[12] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_experiment_name_bad[experiment_name4] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_db_type_string_good[mssql] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_db_type_string_bad[MySQL] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_db_type_string_bad[sql] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_experiment_artifact_location_length_good[file:///path/to/artifacts] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_experiment_artifact_location_length_bad[s3://test-bucket/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_list_param_with_valid_list[param_value0] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_list_param_with_none_not_allowed PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_validation.py::test_validate_list_param_with_invalid_type[param_name-value-str] PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/utils/test_yaml_utils.py::test_yaml_write_sorting PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/webhooks/test_delivery.py::test_deliver_webhook_handles_exception_for_sql_store PASSED | MEM 5.2/16.0 GB | DISK 8.5/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_model_version_created PASSED | MEM 5.8/16.0 GB | DISK 9.0/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_model_version_alias_created PASSED | MEM 5.8/16.0 GB | DISK 9.0/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_webhook_with_wrong_secret PASSED | MEM 5.7/16.0 GB | DISK 8.6/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_webhook_test_secure_endpoint PASSED | MEM 5.7/16.0 GB | DISK 8.6/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_webhook_test_with_wrong_secret PASSED | MEM 5.7/16.0 GB | DISK 8.6/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_prompt_created PASSED | MEM 5.5/16.0 GB | DISK 9.2/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_prompt_tag_deleted PASSED | MEM 5.6/16.0 GB | DISK 9.2/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_prompt_alias_created PASSED | MEM 5.6/16.0 GB | DISK 9.2/150.0 GB [ 99%]

tests/webhooks/test_e2e.py::test_prompt_webhook_test_endpoint PASSED | MEM 4.7/16.0 GB | DISK 8.8/150.0 GB [100%]

================================== FAILURES ===================================

_________________ test_logging_many_traces_in_single_request __________________

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

method = 'POST', url = '/v1/traces'

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

timeout = Timeout(connect=10, read=10, total=None), chunked = False

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

preload_content = False, decode_content = False, enforce_content_length = True

def _make_request(

self,

conn: BaseHTTPConnection,

method: str,

url: str,

body: _TYPE_BODY | None = None,

headers: typing.Mapping[str, str] | None = None,

retries: Retry | None = None,

timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,

chunked: bool = False,

response_conn: BaseHTTPConnection | None = None,

preload_content: bool = True,

decode_content: bool = True,

enforce_content_length: bool = True,

) -> BaseHTTPResponse:

"""

Perform a request on a given urllib connection object taken from our

pool.

:param conn:

a connection from one of our connection pools

:param method:

HTTP request method (such as GET, POST, PUT, etc.)

:param url:

The URL to perform the request on.

:param body:

Data to send in the request body, either :class:`str`, :class:`bytes`,

an iterable of :class:`str`/:class:`bytes`, or a file-like object.

:param headers:

Dictionary of custom headers to send, such as User-Agent,

If-None-Match, etc. If None, pool headers are used. If provided,

these headers completely replace any pool-specific headers.

:param retries:

Configure the number of retries to allow before raising a

:class:`~urllib3.exceptions.MaxRetryError` exception.

Pass ``None`` to retry until you receive a response. Pass a

:class:`~urllib3.util.retry.Retry` object for fine-grained control

over different types of retries.

Pass an integer number to retry connection errors that many times,

but no other types of errors. Pass zero to never retry.

If ``False``, then retries are disabled and any exception is raised

immediately. Also, instead of raising a MaxRetryError on redirects,

the redirect response will be returned.

:type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

:param timeout:

If specified, overrides the default timeout for this one

request. It may be a float (in seconds) or an instance of

:class:`urllib3.util.Timeout`.

:param chunked:

If True, urllib3 will send the body using chunked transfer

encoding. Otherwise, urllib3 will send the body using the standard

content-length form. Defaults to False.

:param response_conn:

Set this to ``None`` if you will handle releasing the connection or

set the connection to have the response release it.

:param preload_content:

If True, the response's body will be preloaded during construction.

:param decode_content:

If True, will attempt to decode the body based on the

'content-encoding' header.

:param enforce_content_length:

Enforce content length checking. Body returned by server must match

value of Content-Length header, if present. Otherwise, raise error.

"""

self.num_requests += 1

timeout_obj = self._get_timeout(timeout)

timeout_obj.start_connect()

conn.timeout = Timeout.resolve_default_timeout(timeout_obj.connect_timeout)

try:

# Trigger any extra validation we need to do.

try:

self._validate_conn(conn)

except (SocketTimeout, BaseSSLError) as e:

self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)

raise

# _validate_conn() starts the connection to an HTTPS proxy

# so we need to wrap errors with 'ProxyError' here too.

except (

OSError,

NewConnectionError,

TimeoutError,

BaseSSLError,

CertificateError,

SSLError,

) as e:

new_e: Exception = e

if isinstance(e, (BaseSSLError, CertificateError)):

new_e = SSLError(e)

# If the connection didn't successfully connect to it's proxy

# then there

if isinstance(

new_e, (OSError, NewConnectionError, TimeoutError, SSLError)

) and (conn and conn.proxy and not conn.has_connected_to_proxy):

new_e = _wrap_proxy_error(new_e, conn.proxy.scheme)

raise new_e

# conn.request() calls http.client.*.request, not the method in

# urllib3.request. It also calls makefile (recv) on the socket.

try:

conn.request(

method,

url,

body=body,

headers=headers,

chunked=chunked,

preload_content=preload_content,

decode_content=decode_content,

enforce_content_length=enforce_content_length,

)

# We are swallowing BrokenPipeError (errno.EPIPE) since the server is

# legitimately able to close the connection after sending a valid response.

# With this behaviour, the received response is still readable.

except BrokenPipeError:

pass

except OSError as e:

# MacOS/Linux

# EPROTOTYPE and ECONNRESET are needed on macOS

# [https://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/](https://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/)

# Condition changed later to emit ECONNRESET instead of only EPROTOTYPE.

if e.errno != errno.EPROTOTYPE and e.errno != errno.ECONNRESET:

raise

# Reset the timeout for the recv() on the socket

read_timeout = timeout_obj.read_timeout

if not conn.is_closed:

# In Python 3 socket.py will catch EAGAIN and return None when you

# try and read into the file pointer created by http.client, which

# instead raises a BadStatusLine exception. Instead of catching

# the exception and assuming all BadStatusLine exceptions are read

# timeouts, check for a zero timeout before making the request.

if read_timeout == 0:

raise ReadTimeoutError(

self, url, f"Read timed out. (read timeout={read_timeout})"

)

conn.timeout = read_timeout

# Receive the response from the server

try:

> response = conn.getresponse()

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

chunked = False

conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

decode_content = False

enforce_content_length = True

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

method = 'POST'

preload_content = False

read_timeout = 10

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

.venv\lib\site-packages\urllib3\connectionpool.py:534:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

.venv\lib\site-packages\urllib3\connection.py:516: in getresponse

httplib_response = super().getresponse()

HTTPResponse = <class 'urllib3.response.HTTPResponse'>

__class__ = <class 'urllib3.connection.HTTPConnection'>

_shutdown = <built-in method shutdown of socket object at 0x00000241119F9060>

resp_options = _ResponseOptions(request_method='POST', request_url='/v1/traces', preload_content=False, decode_content=False, enforce_content_length=True)

self = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\http\client.py:1375: in getresponse

response.begin()

response = <http.client.HTTPResponse object at 0x0000024111A3A200>

self = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\http\client.py:318: in begin

version, status, reason = self._read_status()

self = <http.client.HTTPResponse object at 0x0000024111A3A200>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\http\client.py:279: in _read_status

line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")

self = <http.client.HTTPResponse object at 0x0000024111A3A200>

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <socket.SocketIO object at 0x0000024111A38A00>

b = <memory at 0x0000024113090040>

def readinto(self, b):

"""Read up to len(b) bytes into the writable buffer *b* and return

the number of bytes read. If the socket is non-blocking and no bytes

are available, None is returned.

If *b* is non-empty, a 0 return value indicates that the connection

was shutdown at the other end.

"""

self._checkClosed()

self._checkReadable()

if self._timeout_occurred:

raise OSError("cannot read from timed out object")

while True:

try:

> return self._sock.recv_into(b)

E TimeoutError: timed out

b = <memory at 0x0000024113090040>

self = <socket.SocketIO object at 0x0000024111A38A00>

C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\socket.py:705: TimeoutError

The above exception was the direct cause of the following exception:

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

request = <PreparedRequest [POST]>, stream = False

timeout = Timeout(connect=10, read=10, total=None), verify = True, cert = None

proxies = OrderedDict()

def send(

self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None

):

"""Sends PreparedRequest object. Returns Response object.

:param request: The :class:`PreparedRequest <PreparedRequest>` being sent.

:param stream: (optional) Whether to stream the request content.

:param timeout: (optional) How long to wait for the server to send

data before giving up, as a float, or a :ref:`(connect timeout,

read timeout) <timeouts>` tuple.

:type timeout: float or tuple or urllib3 Timeout object

:param verify: (optional) Either a boolean, in which case it controls whether

we verify the server's TLS certificate, or a string, in which case it

must be a path to a CA bundle to use

:param cert: (optional) Any user-provided SSL certificate to be trusted.

:param proxies: (optional) The proxies dictionary to apply to the request.

:rtype: requests.Response

"""

try:

conn = self.get_connection_with_tls_context(

request, verify, proxies=proxies, cert=cert

)

except LocationValueError as e:

raise InvalidURL(e, request=request)

self.cert_verify(conn, request.url, verify, cert)

url = self.request_url(request, proxies)

self.add_headers(

request,

stream=stream,

timeout=timeout,

verify=verify,

cert=cert,

proxies=proxies,

)

chunked = not (request.body is None or "Content-Length" in request.headers)

if isinstance(timeout, tuple):

try:

connect, read = timeout

timeout = TimeoutSauce(connect=connect, read=read)

except ValueError:

raise ValueError(

f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "

f"or a single float to set both timeouts to the same value."

)

elif isinstance(timeout, TimeoutSauce):

pass

else:

timeout = TimeoutSauce(connect=timeout, read=timeout)

try:

> resp = conn.urlopen(

method=request.method,

url=url,

body=request.body,

headers=request.headers,

redirect=False,

assert_same_host=False,

preload_content=False,

decode_content=False,

retries=self.max_retries,

timeout=timeout,

chunked=chunked,

)

cert = None

chunked = False

conn = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

proxies = OrderedDict()

request = <PreparedRequest [POST]>

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

stream = False

timeout = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

verify = True

.venv\lib\site-packages\requests\adapters.py:644:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

.venv\lib\site-packages\urllib3\connectionpool.py:841: in urlopen

retries = retries.increment(

assert_same_host = False

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

body_pos = None

chunked = False

clean_exit = False

conn = None

decode_content = False

destination_scheme = None

err = None

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

http_tunnel_required = False

method = 'POST'

new_e = ReadTimeoutError("HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)")

parsed_url = Url(scheme=None, auth=None, host=None, port=None, path='/v1/traces', query=None, fragment=None)

pool_timeout = None

preload_content = False

redirect = False

release_conn = False

release_this_conn = True

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

response_kw = {}

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

.venv\lib\site-packages\urllib3\util\retry.py:474: in increment

raise reraise(type(error), error, _stacktrace)

_pool = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

_stacktrace = <traceback object at 0x00000241119E3240>

cause = 'unknown'

connect = None

error = ReadTimeoutError("HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)")

method = 'POST'

other = None

read = False

redirect = None

redirect_location = None

response = None

self = Retry(total=0, connect=None, read=False, redirect=None, status=None)

status = None

status_count = None

total = -1

url = '/v1/traces'

.venv\lib\site-packages\urllib3\util\util.py:39: in reraise

raise value

tb = None

tp = <class 'urllib3.exceptions.ReadTimeoutError'>

value = None

.venv\lib\site-packages\urllib3\connectionpool.py:787: in urlopen

response = self._make_request(

assert_same_host = False

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

body_pos = None

chunked = False

clean_exit = False

conn = None

decode_content = False

destination_scheme = None

err = None

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

http_tunnel_required = False

method = 'POST'

new_e = ReadTimeoutError("HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)")

parsed_url = Url(scheme=None, auth=None, host=None, port=None, path='/v1/traces', query=None, fragment=None)

pool_timeout = None

preload_content = False

redirect = False

release_conn = False

release_this_conn = True

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

response_kw = {}

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

.venv\lib\site-packages\urllib3\connectionpool.py:536: in _make_request

self._raise_timeout(err=e, url=url, timeout_value=read_timeout)

body = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

chunked = False

conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

decode_content = False

enforce_content_length = True

headers = {'User-Agent': 'python-requests/2.32.5', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1', 'Content-Length': '1580'}

method = 'POST'

preload_content = False

read_timeout = 10

response_conn = <urllib3.connection.HTTPConnection object at 0x0000024111A3B5E0>

retries = Retry(total=0, connect=None, read=False, redirect=None, status=None)

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout = Timeout(connect=10, read=10, total=None)

timeout_obj = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

err = TimeoutError('timed out'), url = '/v1/traces', timeout_value = 10

def _raise_timeout(

self,

err: BaseSSLError | OSError | SocketTimeout,

url: str,

timeout_value: _TYPE_TIMEOUT | None,

) -> None:

"""Is the error actually a timeout? Will raise a ReadTimeout or pass"""

if isinstance(err, SocketTimeout):

> raise ReadTimeoutError(

self, url, f"Read timed out. (read timeout={timeout_value})"

) from err

E urllib3.exceptions.ReadTimeoutError: HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)

err = TimeoutError('timed out')

self = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

timeout_value = 10

url = '/v1/traces'

.venv\lib\site-packages\urllib3\connectionpool.py:367: ReadTimeoutError

During handling of the above exception, another exception occurred:

mlflow_server = '[http://127.0.0.1:62039](http://127.0.0.1:62039)'

def test_logging_many_traces_in_single_request(mlflow_server: str):

mlflow.set_tracking_uri(mlflow_server)

experiment = mlflow.set_experiment("otel-many-traces-test")

experiment_id = experiment.experiment_id

# Create a request with 15 different traces (exceeds the 10 thread pool limit)

request = ExportTraceServiceRequest()

num_traces = 15

for trace_num in range(num_traces):

span = OTelProtoSpan()

trace_id_hex = f"{trace_num + 1000:016x}" + "0" * 16

span.trace_id = bytes.fromhex(trace_id_hex)

span.span_id = bytes.fromhex(f"{trace_num + 1000:08x}" + "0" * 8)

span.name = f"many-traces-test-span-{trace_num}"

span.start_time_unix_nano = 1000000000 + trace_num * 1000

span.end_time_unix_nano = 2000000000 + trace_num * 1000

scope = InstrumentationScope()

scope.name = "many-traces-test-scope"

scope_spans = ScopeSpans()

scope_spans.scope.CopyFrom(scope)

scope_spans.spans.append(span)

resource = Resource()

resource_spans = ResourceSpans()

resource_spans.resource.CopyFrom(resource)

resource_spans.scope_spans.append(scope_spans)

request.resource_spans.append(resource_spans)

# Send the request and measure response time

> requests.post(

f"{mlflow_server}/v1/traces",

data=request.SerializeToString(),

headers={

"Content-Type": "application/x-protobuf",

MLFLOW_EXPERIMENT_ID_HEADER: experiment_id,

},

timeout=10,

)

experiment = <Experiment: artifact_location='file:///C:/Users/runneradmin/AppData/Local/Temp/pytest-of-runneradmin/pytest-0/test_lo...95, experiment_id='1', last_update_time=1767752261095, lifecycle_stage='active', name='otel-many-traces-test', tags={}>

experiment_id = '1'

mlflow_server = '[http://127.0.0.1:62039](http://127.0.0.1:62039)'

num_traces = 15

request = resource_spans {

resource {

}

scope_spans {

scope {

name: "many-traces-test-scope"

}

spans {

...me: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

}

}

}

resource =

resource_spans = resource {

}

scope_spans {

scope {

name: "many-traces-test-scope"

}

spans {

trace_id: "\000\000\000\000\...00"

name: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

}

}

scope = name: "many-traces-test-scope"

scope_spans = scope {

name: "many-traces-test-scope"

}

spans {

trace_id: "\000\000\000\000\000\000\003\366\000\000\000\000\000\0...\000\000\000"

name: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

}

span = trace_id: "\000\000\000\000\000\000\003\366\000\000\000\000\000\000\000\000"

span_id: "\000\000\003\366\000\000\000\000"

name: "many-traces-test-span-14"

start_time_unix_nano: 1000014000

end_time_unix_nano: 2000014000

trace_id_hex = '00000000000003f60000000000000000'

trace_num = 14

tests\tracing\test_otel_logging.py:476:

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

.venv\lib\site-packages\requests\api.py:115: in post

return request("post", url, data=data, json=json, **kwargs)

data = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

json = None

kwargs = {'headers': {'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1'}, 'timeout': 10}

url = '[http://127.0.0.1:62039/v1/traces](http://127.0.0.1:62039/v1/traces)'

.venv\lib\site-packages\requests\api.py:59: in request

return session.request(method=method, url=url, **kwargs)

kwargs = {'data': b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00...00', 'headers': {'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1'}, 'json': None, 'timeout': 10}

method = 'post'

session = <requests.sessions.Session object at 0x0000024111A078E0>

url = '[http://127.0.0.1:62039/v1/traces](http://127.0.0.1:62039/v1/traces)'

.venv\lib\site-packages\requests\sessions.py:589: in request

resp = self.send(prep, **send_kwargs)

allow_redirects = True

auth = None

cert = None

cookies = None

data = b'\ng\n\x00\x12c\n\x18\n\x16many-traces-test-scope\x12G\n\x10\x00\x00\x00\x00\x00\x00\x03\xe8\x00\x00\x00\x00\x00\x00\...\x00\x00\x03\xf6\x00\x00\x00\x00*\x18many-traces-test-span-149\xb0\x00\x9b;\x00\x00\x00\x00A\xb0\xca5w\x00\x00\x00\x00'

files = None

headers = {'Content-Type': 'application/x-protobuf', 'x-mlflow-experiment-id': '1'}

hooks = None

json = None

method = 'post'

params = None

prep = <PreparedRequest [POST]>

proxies = {}

req = <Request [POST]>

self = <requests.sessions.Session object at 0x0000024111A078E0>

send_kwargs = {'allow_redirects': True, 'cert': None, 'proxies': OrderedDict(), 'stream': False, ...}

settings = {'cert': None, 'proxies': OrderedDict(), 'stream': False, 'verify': True}

stream = None

timeout = 10

url = '[http://127.0.0.1:62039/v1/traces](http://127.0.0.1:62039/v1/traces)'

verify = None

.venv\lib\site-packages\requests\sessions.py:703: in send

r = adapter.send(request, **kwargs)

adapter = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

allow_redirects = True

hooks = {'response': []}

kwargs = {'cert': None, 'proxies': OrderedDict(), 'stream': False, 'timeout': 10, ...}

request = <PreparedRequest [POST]>

self = <requests.sessions.Session object at 0x0000024111A078E0>

start = 1666.4083634

stream = False

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

request = <PreparedRequest [POST]>, stream = False

timeout = Timeout(connect=10, read=10, total=None), verify = True, cert = None

proxies = OrderedDict()

def send(

self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None

):

"""Sends PreparedRequest object. Returns Response object.

:param request: The :class:`PreparedRequest <PreparedRequest>` being sent.

:param stream: (optional) Whether to stream the request content.

:param timeout: (optional) How long to wait for the server to send

data before giving up, as a float, or a :ref:`(connect timeout,

read timeout) <timeouts>` tuple.

:type timeout: float or tuple or urllib3 Timeout object

:param verify: (optional) Either a boolean, in which case it controls whether

we verify the server's TLS certificate, or a string, in which case it

must be a path to a CA bundle to use

:param cert: (optional) Any user-provided SSL certificate to be trusted.

:param proxies: (optional) The proxies dictionary to apply to the request.

:rtype: requests.Response

"""

try:

conn = self.get_connection_with_tls_context(

request, verify, proxies=proxies, cert=cert

)

except LocationValueError as e:

raise InvalidURL(e, request=request)

self.cert_verify(conn, request.url, verify, cert)

url = self.request_url(request, proxies)

self.add_headers(

request,

stream=stream,

timeout=timeout,

verify=verify,

cert=cert,

proxies=proxies,

)

chunked = not (request.body is None or "Content-Length" in request.headers)

if isinstance(timeout, tuple):

try:

connect, read = timeout

timeout = TimeoutSauce(connect=connect, read=read)

except ValueError:

raise ValueError(

f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "

f"or a single float to set both timeouts to the same value."

)

elif isinstance(timeout, TimeoutSauce):

pass

else:

timeout = TimeoutSauce(connect=timeout, read=timeout)

try:

resp = conn.urlopen(

method=request.method,

url=url,

body=request.body,

headers=request.headers,

redirect=False,

assert_same_host=False,

preload_content=False,

decode_content=False,

retries=self.max_retries,

timeout=timeout,

chunked=chunked,

)

except (ProtocolError, OSError) as err:

raise ConnectionError(err, request=request)

except MaxRetryError as e:

if isinstance(e.reason, ConnectTimeoutError):

# TODO: Remove this in 3.0.0: see #2811

if not isinstance(e.reason, NewConnectionError):

raise ConnectTimeout(e, request=request)

if isinstance(e.reason, ResponseError):

raise RetryError(e, request=request)

if isinstance(e.reason, _ProxyError):

raise ProxyError(e, request=request)

if isinstance(e.reason, _SSLError):

# This branch is for urllib3 v1.22 and later.

raise SSLError(e, request=request)

raise ConnectionError(e, request=request)

except ClosedPoolError as e:

raise ConnectionError(e, request=request)

except _ProxyError as e:

raise ProxyError(e)

except (_SSLError, _HTTPError) as e:

if isinstance(e, _SSLError):

# This branch is for urllib3 versions earlier than v1.22

raise SSLError(e, request=request)

elif isinstance(e, ReadTimeoutError):

> raise ReadTimeout(e, request=request)

E requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)

cert = None

chunked = False

conn = <urllib3.connectionpool.HTTPConnectionPool object at 0x0000024111A20F70>

proxies = OrderedDict()

request = <PreparedRequest [POST]>

self = <requests.adapters.HTTPAdapter object at 0x00000241144C6BF0>

stream = False

timeout = Timeout(connect=10, read=10, total=None)

url = '/v1/traces'

verify = True

.venv\lib\site-packages\requests\adapters.py:690: ReadTimeout

---------------------------- Captured stderr setup ----------------------------

2026/01/07 02:17:40 INFO mlflow.store.db.utils: Create SQLAlchemy engine with pool options {'poolclass': <class 'sqlalchemy.pool.impl.NullPool'>}

2026/01/07 02:17:40 INFO alembic.runtime.migration: Context impl SQLiteImpl.

2026/01/07 02:17:40 INFO alembic.runtime.migration: Will assume non-transactional DDL.

2026/01/07 02:17:40 INFO mlflow.store.db.utils: Create SQLAlchemy engine with pool options {'poolclass': <class 'sqlalchemy.pool.impl.NullPool'>}

---------------------------- Captured stderr call -----------------------------

2026/01/07 02:17:41 INFO mlflow.tracking.fluent: Experiment with name 'otel-many-traces-test' does not exist. Creating a new experiment.

-------------------------- Captured stderr teardown ---------------------------

2026/01/07 02:17:58 INFO mlflow.tracking.fluent: Active model is cleared

============================== warnings summary ===============================

mlflow\pyfunc\utils\data_validation.py:186

mlflow\pyfunc\utils\data_validation.py:186

mlflow\pyfunc\utils\data_validation.py:186

tests/store/model_registry/test_file_store.py::test_create_model_version_with_model_id_and_no_run_id

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

tests/tracking/fluent/test_fluent.py::test_last_logged_model_log_model

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:186: UserWarning: Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check [https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel](https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel) for more details.

color_warning(

tests\metrics\test_metric_definitions.py:38

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:38: FutureWarning: ``mlflow.metrics.ari_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

ari_grade_level(),

tests\metrics\test_metric_definitions.py:39

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:39: FutureWarning: ``mlflow.metrics.exact_match`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

exact_match(),

tests\metrics\test_metric_definitions.py:40

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:40: FutureWarning: ``mlflow.metrics.flesch_kincaid_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

flesch_kincaid_grade_level(),

tests\metrics\test_metric_definitions.py:41

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:41: FutureWarning: ``mlflow.metrics.rouge1`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rouge1(),

tests\metrics\test_metric_definitions.py:42

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:42: FutureWarning: ``mlflow.metrics.rouge2`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rouge2(),

tests\metrics\test_metric_definitions.py:43

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:43: FutureWarning: ``mlflow.metrics.rougeL`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rougeL(),

tests\metrics\test_metric_definitions.py:44

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:44: FutureWarning: ``mlflow.metrics.rougeLsum`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

rougeLsum(),

tests\metrics\test_metric_definitions.py:45

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:45: FutureWarning: ``mlflow.metrics.toxicity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

toxicity(),

tests\metrics\test_metric_definitions.py:46

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:46: FutureWarning: ``mlflow.metrics.bleu`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

bleu(),

.venv\lib\site-packages\google\api_core\_python_version_support.py:266

D:\a\mlflow\mlflow\.venv\lib\site-packages\google\api_core\_python_version_support.py:266: FutureWarning: You are using a Python version (3.10.11) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.

warnings.warn(message, FutureWarning)

.venv\lib\site-packages\starlette\middleware\wsgi.py:14

D:\a\mlflow\mlflow\.venv\lib\site-packages\starlette\middleware\wsgi.py:14: DeprecationWarning: starlette.middleware.wsgi is deprecated and will be removed in a future release. Please refer to [https://github.com/abersheeran/a2wsgi](https://github.com/abersheeran/a2wsgi) as a replacement.

warnings.warn(

tests/data/test_artifact_dataset_sources.py::test_expected_artifact_dataset_sources_are_registered_and_resolvable[file:///tmp/path/to/my/local/directory-local-LocalArtifactDatasetSource]

tests/data/test_artifact_dataset_sources.py::test_to_and_from_json[file:///tmp/path/to/my/local/directory-local]

tests/data/test_artifact_dataset_sources.py::test_load_makes_expected_mlflow_artifacts_download_call[file:///tmp/path/to/my/local/directory-local]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'file:///tmp/path/to/my/local/directory'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py: 4 warnings

tests/data/test_meta_dataset.py: 1 warning

tests/data/test_pandas_dataset.py: 1 warning

tests/data/test_polars_dataset.py: 1 warning

tests/data/test_tensorflow_dataset.py: 2 warnings

tests/tracking/fluent/test_fluent.py: 1 warning

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: The specified dataset source can be interpreted in multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource. MLflow will assume that this is a LocalArtifactDatasetSource source.

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_expected_artifact_dataset_sources_are_registered_and_resolvable[wasbs://user@host.blob.core.windows.net/dir-wasbs-AzureBlobArtifactDatasetSource]

tests/data/test_artifact_dataset_sources.py::test_to_and_from_json[wasbs://user@host.blob.core.windows.net/dir-wasbs]

tests/data/test_artifact_dataset_sources.py::test_load_makes_expected_mlflow_artifacts_download_call[wasbs://user@host.blob.core.windows.net/dir-wasbs]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'wasbs://user@host.blob.core.windows.net/dir'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_expected_artifact_dataset_sources_are_registered_and_resolvable[hdfs://host_name:8020/hdfs/path/to/my/file.txt-hdfs-HdfsArtifactDatasetSource]

tests/data/test_artifact_dataset_sources.py::test_to_and_from_json[hdfs://host_name:8020/hdfs/path/to/my/file.txt-hdfs]

tests/data/test_artifact_dataset_sources.py::test_load_makes_expected_mlflow_artifacts_download_call[hdfs://host_name:8020/hdfs/path/to/my/file.txt-hdfs]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'hdfs://host_name:8020/hdfs/path/to/my/file.txt'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_local_load[dst]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_local_load_dst_0\myfile.txt'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_artifact_dataset_sources.py::test_local_load[dst]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_local_load_dst_0\mydir'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_dataset_source.py::test_get_source_obtains_expected_code_source

tests/data/test_huggingface_dataset_and_source.py::test_dataset_conversion_to_json

tests/data/test_meta_dataset.py::test_create_meta_dataset_from_dataset

tests/data/test_pandas_dataset.py::test_from_pandas_file_system_datasource

tests/telemetry/test_tracked_events.py::test_evaluate

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

D:\a\mlflow\mlflow\mlflow\types\utils.py:452: UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <[https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values](https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values)>`_ for more details.

warnings.warn(

tests/data/test_dataset_source_registry.py::test_resolve_dataset_source_maintains_consistent_order_and_uses_last_registered_match

D:\a\mlflow\mlflow\tests\data\test_dataset_source_registry.py:108: UserWarning: The specified dataset source can be interpreted in multiple ways: SampleDatasetSource, SampleDatasetSourceCopy1, SampleDatasetSourceCopy2. MLflow will assume that this is a SampleDatasetSourceCopy2 source.

source1 = registry1.resolve("test:/" + str(tmp_path))

tests/data/test_dataset_source_registry.py::test_resolve_dataset_source_maintains_consistent_order_and_uses_last_registered_match

D:\a\mlflow\mlflow\tests\data\test_dataset_source_registry.py:115: UserWarning: The specified dataset source can be interpreted in multiple ways: SampleDatasetSource, SampleDatasetSourceCopy2, SampleDatasetSourceCopy1. MLflow will assume that this is a SampleDatasetSourceCopy1 source.

source2 = registry2.resolve("test:/" + str(tmp_path))

tests/data/test_dataset_source_registry.py::test_resolve_dataset_source_maintains_consistent_order_and_uses_last_registered_match

D:\a\mlflow\mlflow\tests\data\test_dataset_source_registry.py:133: UserWarning: The specified dataset source can be interpreted in multiple ways: DummyDatasetSourceCopy, DummyDatasetSource. MLflow will assume that this is a DummyDatasetSource source.

source5 = registry3.resolve("dummy:/" + str(tmp_path))

tests/data/test_meta_dataset.py::test_create_meta_dataset_from_dataset

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for '/tmp/test.csv'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_pandas_dataset.py::test_from_pandas_file_system_datasource

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_from_pandas_file_system_d0\temp.csv'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_polars_dataset.py::test_from_polars_with_targets

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_from_polars_with_targets0\temp.csv'. Exception:

return _dataset_source_registry.resolve(

tests/data/test_tensorflow_dataset.py::test_from_tensorflow_dataset_constructs_expected_dataset

tests/data/test_tensorflow_dataset.py::test_from_tensorflow_tensor_with_targets_constructs_expected_dataset

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'my_source'. Exception:

return _dataset_source_registry.resolve(

tests/deployments/databricks/test_databricks.py::test_update_endpoint

D:\a\mlflow\mlflow\tests\deployments\databricks\test_databricks.py:366: FutureWarning: ``mlflow.deployments.databricks.DatabricksDeploymentClient.update_endpoint`` is deprecated. This method will be removed in a future release. Use ``update_endpoint_config, update_endpoint_tags, update_endpoint_rate_limits, or update_endpoint_ai_gateway`` instead.

resp = client.update_endpoint(

tests/entities/test_assessment_source.py::test_assessment_source_type_validation

D:\a\mlflow\mlflow\mlflow\entities\assessment_source.py:169: FutureWarning: AI_JUDGE is deprecated. Use LLM_JUDGE instead.

warnings.warn(

tests/entities/test_trace_data.py::test_intermediate_outputs_from_attribute

D:\a\mlflow\mlflow\tests\entities\test_trace_data.py:168: FutureWarning: ``mlflow.entities.trace_data.TraceData.intermediate_outputs`` is deprecated since 3.6.0. This method will be removed in a future release. Use ``trace.search_spans(name=...)`` instead.

assert trace.data.intermediate_outputs == intermediate_outputs

tests/entities/test_trace_location.py::test_trace_location_mismatch

D:\a\mlflow\mlflow\tests\entities\test_trace_location.py:58: FutureWarning: ``mlflow.entities.trace_location.InferenceTableLocation`` is deprecated since 3.7.0. This method will be removed in a future release.

inference_table=InferenceTableLocation(full_table_name="a.b.c"),

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_correct_response

tests/metrics/genai/test_genai_metrics.py::test_log_make_genai_metric_fn_args

tests/metrics/genai/test_genai_metrics.py::test_genai_metrics_callable

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:148: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_correct_response

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:188: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

custom_metric = make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_malformed_input_raises_exception

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:397: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_similarity_metric = answer_similarity()

tests/metrics/genai/test_genai_metrics.py::test_malformed_input_raises_exception

tests/metrics/genai/test_genai_metrics.py::test_similarity_metric[parameters1-None-None]

tests/metrics/genai/test_genai_metrics.py::test_genai_metric_with_custom_chat_endpoint[True]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:85: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_throws_if_grading_context_cols_wrong[good_column-bad_column]

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_throws_if_grading_context_cols_wrong[grading_cols3-example_context_cols3]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:547: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_similarity_metric[parameters1-None-None]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:647: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

similarity_metric = answer_similarity(

tests/metrics/genai/test_genai_metrics.py::test_similarity_metric[parameters1-None-None]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:719: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_similarity(

tests/metrics/genai/test_genai_metrics.py::test_answer_correctness_metric

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:806: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_correctness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_correctness_metric = answer_correctness()

tests/metrics/genai/test_genai_metrics.py::test_answer_correctness_metric

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[answer_correctness]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:178: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_answer_correctness_metric

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:871: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_correctness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

answer_correctness(metric_version="non-existent-version")

tests/metrics/genai/test_genai_metrics.py::test_make_genai_metric_metric_details

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1023: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

custom_metric = make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_make_custom_judge_prompt_genai_metric

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1089: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric_from_prompt`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

custom_judge_prompt_metric = make_genai_metric_from_prompt(

tests/metrics/genai/test_genai_metrics.py::test_metric_metadata_on_prebuilt_genai_metrics[faithfulness]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1234: FutureWarning: ``mlflow.metrics.genai.metric_definitions.faithfulness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric = metric_fn(metric_metadata={"metadata_field": "metadata_value"})

tests/metrics/genai/test_genai_metrics.py::test_metric_metadata_on_prebuilt_genai_metrics[faithfulness]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:266: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/genai/test_genai_metrics.py::test_genai_metric_with_custom_chat_endpoint[True]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1335: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_similarity`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

similarity_metric = answer_similarity(

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[answer_correctness]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1392: FutureWarning: ``mlflow.metrics.genai.metric_definitions.answer_correctness`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric_fn(parameters={"temperature": 0.1})

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[relevance]

D:\a\mlflow\mlflow\tests\metrics\genai\test_genai_metrics.py:1392: FutureWarning: ``mlflow.metrics.genai.metric_definitions.relevance`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric_fn(parameters={"temperature": 0.1})

tests/metrics/genai/test_genai_metrics.py::test_metric_parameters_on_prebuilt_genai_metrics[relevance]

D:\a\mlflow\mlflow\mlflow\metrics\genai\metric_definitions.py:441: FutureWarning: ``mlflow.metrics.genai.genai_metric.make_genai_metric`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

return make_genai_metric(

tests/metrics/test_metric_definitions.py::test_flesch_kincaid_grade_level

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:90: FutureWarning: ``mlflow.metrics.flesch_kincaid_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = flesch_kincaid_grade_level().eval_fn(predictions, None, {})

tests/metrics/test_metric_definitions.py::test_flesch_kincaid_grade_level

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:95: FutureWarning: ``mlflow.metrics.flesch_kincaid_grade_level`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert flesch_kincaid_grade_level()(predictions=predictions) == result

tests/metrics/test_metric_definitions.py::test_rouge1

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:133: FutureWarning: ``mlflow.metrics.rouge1`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = rouge1().eval_fn(predictions, targets, {})

tests/metrics/test_metric_definitions.py::test_rouge1

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:139: FutureWarning: ``mlflow.metrics.rouge1`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert rouge1()(predictions=predictions, targets=targets) == result

tests/metrics/test_metric_definitions.py::test_rougeLsum

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:169: FutureWarning: ``mlflow.metrics.rougeLsum`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = rougeLsum().eval_fn(predictions, targets, {})

tests/metrics/test_metric_definitions.py::test_rougeLsum

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:175: FutureWarning: ``mlflow.metrics.rougeLsum`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert rougeLsum()(predictions=predictions, targets=targets) == result

tests/metrics/test_metric_definitions.py::test_recall_at_k

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:287: FutureWarning: ``mlflow.metrics.recall_at_k`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

result = recall_at_k(4).eval_fn(predictions, targets)

tests/metrics/test_metric_definitions.py::test_recall_at_k

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:295: FutureWarning: ``mlflow.metrics.recall_at_k`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

assert recall_at_k(4)(predictions=predictions, targets=targets) == result

tests/metrics/test_metric_definitions.py::test_builtin_metric_call_signature

D:\a\mlflow\mlflow\tests\metrics\test_metric_definitions.py:391: FutureWarning: ``mlflow.metrics.ndcg_at_k`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

metric = ndcg_at_k(3)

tests/store/model_registry/test_rest_store_webhooks.py::test_create_webhook

D:\a\mlflow\mlflow\.venv\lib\site-packages\websockets\legacy\__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see [https://websockets.readthedocs.io/en/stable/howto/upgrade.html](https://websockets.readthedocs.io/en/stable/howto/upgrade.html) for upgrade instructions

warnings.warn( # deprecated in 14.0 - 2024-11-09

tests/store/model_registry/test_rest_store_webhooks.py::test_create_webhook

D:\a\mlflow\mlflow\.venv\lib\site-packages\uvicorn\protocols\websockets\websockets_impl.py:17: DeprecationWarning: websockets.server.WebSocketServerProtocol is deprecated

from websockets.server import WebSocketServerProtocol

tests/store/tracking/test_sqlalchemy_store.py::test_run_needs_uuid

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:187: SAWarning: Column 'runs.run_uuid' is marked as a member of the primary key for table 'runs', but has no Python-side or server-side default generator indicated, nor does it indicate 'autoincrement=True' or 'nullable=True', and no explicit value is passed. Primary key columns typically may not store NULL.

session.commit()

tests/store/tracking/test_sqlalchemy_store.py::test_delete_traces_with_max_count

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

tests/tracing/test_provider.py::test_disable_enable_tracing

tests/tracing/utils/test_copy.py::test_copy_trace_missing_info

D:\a\mlflow\mlflow\mlflow\store\tracking\sqlalchemy_store.py:3002: SAWarning: Coercing Subquery object into a select() for use in IN(); please pass a select() construct explicitly

SqlTraceInfo.request_id.in_(

tests/telemetry/test_tracked_events.py::test_evaluate

D:\a\mlflow\mlflow\tests\telemetry\test_tracked_events.py:349: FutureWarning: ``mlflow.metrics.latency`` is deprecated since 3.4.0. Use the new GenAI evaluation functionality instead. See [https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/](https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/) for the migration guide.

extra_metrics=[mlflow.metrics.latency()],

tests/telemetry/test_tracked_events.py::test_invoke_custom_judge_model[endpoints:/my-endpoint-endpoints-True-False]

D:\a\mlflow\mlflow\tests\telemetry\test_tracked_events.py:992: FutureWarning: The legacy provider 'endpoints' is deprecated and will be removed in a future release. Please update your code to use the 'databricks' provider instead.

invoke_judge_model(

tests/tracing/test_fluent.py::test_trace_with_experiment_id_issue_warning_when_not_root_span

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:652: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

@mlflow.trace(trace_destination=MlflowExperiment(exp_1))

tests/tracing/test_fluent.py::test_trace_with_experiment_id_issue_warning_when_not_root_span

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:656: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

@mlflow.trace(trace_destination=MlflowExperiment(exp_1))

tests/tracing/test_fluent.py::test_search_traces[list]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:913: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_fluent.py::test_search_traces_with_pagination

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:974: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(experiment_ids=["1"])

tests/tracing/test_fluent.py::test_search_traces_with_non_dict_span_inputs_outputs

tests/tracing/test_fluent.py::test_search_traces_with_non_existent_field

tests/tracing/test_fluent.py::test_search_traces_invalid_extract_fields[extract_fields0]

D:\a\mlflow\mlflow\mlflow\utils\annotations.py:260: FutureWarning: The `extract_fields` parameter is deprecated and will be removed in a future version.

return func(*args, **kwargs)

tests/tracing/test_fluent.py::test_update_current_trace_should_not_raise_during_model_logging

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:155: FutureWarning: Model's `predict` method contains invalid parameters: {'model_inputs'}. Only the following parameter names are allowed: context, model_input, and params. Note that invalid parameters will no longer be permitted in future versions.

param_names = _check_func_signature(func, "predict")

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-outputs0-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-Yes-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-None-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[None-outputs0-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[None-Yes-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[None-None-inputs0]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2134: FutureWarning: ``mlflow.tracing.fluent.log_trace`` is deprecated since 3.6.0. This method will be removed in a future release.

mlflow.log_trace(

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-outputs0-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-Yes-inputs0]

tests/tracing/test_fluent.py::test_log_trace_success[intermediate_outputs0-None-inputs0]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2153: FutureWarning: ``mlflow.entities.trace_data.TraceData.intermediate_outputs`` is deprecated since 3.6.0. This method will be removed in a future release. Use ``trace.search_spans(name=...)`` instead.

assert trace.data.intermediate_outputs == intermediate_outputs

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2175: FutureWarning: Parameter 'request_id' is deprecated. Please use 'trace_id' instead.

mlflow.set_trace_tag(request_id=trace_id, key="key3", value="value3")

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2176: FutureWarning: Parameter 'request_id' is deprecated. Please use 'trace_id' instead.

trace = mlflow.get_trace(request_id=trace_id)

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2179: FutureWarning: Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

mlflow.delete_trace_tag(request_id=trace_id, key="key3")

tests/tracing/test_fluent.py::test_set_delete_trace_tag

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2180: FutureWarning: Parameter 'request_id' is deprecated. Please use 'trace_id' instead.

trace = mlflow.get_trace(request_id=trace_id)

tests/tracing/test_fluent.py::test_set_destination_in_threads[True]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2279: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

set_destination(MlflowExperiment(experiment_id1))

tests/tracing/test_fluent.py::test_set_destination_in_threads[True]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2272: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

set_destination(MlflowExperiment(experiment_id), context_local=True)

tests/tracing/test_fluent.py::test_set_destination_in_async_contexts[False]

D:\a\mlflow\mlflow\tests\tracing\test_fluent.py:2340: FutureWarning: ``mlflow.tracing.destination.MlflowExperiment`` is deprecated since 3.5.0. This method will be removed in a future release. Use ``mlflow.entities.trace_location.MlflowExperimentLocation`` instead.

set_destination(MlflowExperiment(experiment_id), context_local=True)

tests/tracing/test_otel_loading.py::test_get_trace_for_otel_sent_span[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:110: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_get_trace_with_otel_span_events[True]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:198: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_get_trace_with_otel_span_status[True]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:238: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_set_trace_tag_on_otel_trace[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:265: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_log_feedback_on_otel_trace[True]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:331: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_multiple_assessments_on_otel_trace[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:390: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_multiple_assessments_on_otel_trace[False]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:445: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tagged_traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_span_inputs_outputs_translation[True-GenAiTranslator]

tests/tracing/test_otel_loading.py::test_span_inputs_outputs_translation[False-GenAiTranslator]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:508: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_span_token_usage_translation[True-GenAiTranslator]

tests/tracing/test_otel_loading.py::test_span_token_usage_translation[False-GenAiTranslator]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:539: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_loading.py::test_aggregated_token_usage_from_multiple_spans[True-GenAiTranslator]

tests/tracing/test_otel_loading.py::test_aggregated_token_usage_from_multiple_spans[False-GenAiTranslator]

D:\a\mlflow\mlflow\tests\tracing\test_otel_loading.py:582: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_otel_logging.py::test_otel_client_sends_spans_to_mlflow_database

D:\a\mlflow\mlflow\tests\tracing\test_otel_logging.py:127: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

traces = mlflow.search_traces(

tests/tracing/test_tracing_client.py::test_tracing_client_link_prompt_versions_to_trace

D:\a\mlflow\mlflow\tests\tracing\test_tracing_client.py:52: FutureWarning: The `mlflow.register_prompt` API is moved to the `mlflow.genai` namespace. Please use `mlflow.genai.register_prompt` instead. The original API will be removed in the future release.

prompt_version = mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:183: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

session.execute(sql.text("PRAGMA foreign_keys = ON;"))

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:184: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

session.execute(sql.text("PRAGMA busy_timeout = 20000;"))

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

D:\a\mlflow\mlflow\mlflow\store\db\utils.py:185: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

session.execute(sql.text("PRAGMA case_sensitive_like = true;"))

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store[sqlite]

tests/tracking/_tracking_service/test_utils.py::test_get_store_sqlalchemy_store_with_artifact_uri[mysql]

D:\a\mlflow\mlflow\mlflow\store\tracking\sqlalchemy_store.py:486: SAWarning: The engine provided as bind produced a connection that is already in a transaction. This is usually caused by a core event, such as 'engine_connect', that has left a transaction open. The effective join transaction mode used by this session is 'create_savepoint'. To silence this warning, do not leave transactions open

.one_or_none()

tests/tracking/fluent/test_fluent.py::test_log_input_polars

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning: Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_log_input_polars0\temp.csv'. Exception:

return _dataset_source_registry.resolve(

tests/tracking/test_client.py::test_start_and_end_trace_does_not_log_trace_when_disabled[file-True]

tests/tracking/test_client.py::test_start_and_end_trace_does_not_log_trace_when_disabled[sqlalchemy-False]

D:\a\mlflow\mlflow\tests\tracking\test_client.py:1028: FutureWarning: Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

assert client.search_traces(experiment_ids=[experiment_id]) == []

tests/tracking/test_client.py::test_transition_model_version_stage

D:\a\mlflow\mlflow\tests\tracking\test_client.py:1449: FutureWarning: ``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

actual_result = MlflowClient(registry_uri="sqlite:///somedb.db").transition_model_version_stage(

tests/tracking/test_client.py::test_crud_prompts[file]

D:\a\mlflow\mlflow\mlflow\prompt\registry_utils.py:209: FutureWarning: The `mlflow.load_prompt` API is moved to the `mlflow.genai` namespace. Please use `mlflow.genai.load_prompt` instead. The original API will be removed in the future release.

return func(*args, **kwargs)

tests/tracking/test_client.py::test_block_handling_prompt_with_model_apis[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_client.py:2363: FutureWarning: ``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

api(*args)

tests/tracking/test_client.py::test_block_handling_prompt_with_model_apis[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_client.py:2363: FutureWarning: ``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

api(*args)

tests/tracking/test_model_registry.py::test_search_registered_model_flow_paginated[file-None-<lambda>-100]

D:\a\mlflow\mlflow\.venv\lib\site-packages\_pytest\unraisableexception.py:65: PytestUnraisableExceptionWarning:

Exception ignored in: <function Variable.__del__ at 0x0000024111DE3760>

Traceback (most recent call last):

File "C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\tkinter\__init__.py", line 388, in __del__

if self._tk.getboolean(self._tk.call("info", "exists", self._name)):

RuntimeError: main thread is not in main loop

Enable tracemalloc to get traceback where the object was allocated.

See [https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings](https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings) for more info.

tests/tracking/test_model_registry.py::test_search_registered_model_flow_paginated[file-None-<lambda>-100]

D:\a\mlflow\mlflow\.venv\lib\site-packages\_pytest\unraisableexception.py:65: PytestUnraisableExceptionWarning:

Exception ignored in: <function Image.__del__ at 0x000002411A89F400>

Traceback (most recent call last):

File "C:\hostedtoolcache\windows\Python\3.10.11\x64\lib\tkinter\__init__.py", line 4056, in __del__

self.tk.call('image', 'delete', self.name)

RuntimeError: main thread is not in main loop

Enable tracemalloc to get traceback where the object was allocated.

See [https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings](https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings) for more info.

tests/tracking/test_model_registry.py::test_update_model_version_flow[file]

D:\a\mlflow\mlflow\tests\tracking\test_model_registry.py:486: FutureWarning:

``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

tests/tracking/test_model_registry.py::test_latest_models[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_model_registry.py:539: FutureWarning:

``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

tests/tracking/test_model_registry.py::test_latest_models[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_model_registry.py:544: FutureWarning:

``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: [https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages)

tests/tracking/test_rest_tracking.py::test_log_input[file]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning:

Failed to determine whether UCVolumeDatasetSource can resolve source information for 'C:\Users\runneradmin\AppData\Local\Temp\pytest-of-runneradmin\pytest-0\test_log_input_file_0\temp.csv'. Exception:

tests/tracking/test_rest_tracking.py::test_log_input[file]

D:\a\mlflow\mlflow\mlflow\data\dataset_source_registry.py:148: UserWarning:

The specified dataset source can be interpreted in multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource. MLflow will assume that this is a LocalArtifactDatasetSource source.

tests/tracking/test_rest_tracking.py::test_log_input[file]

tests/types/test_schema.py::test_schema_inference_on_dictionary_of_strings[data3-schema3]

tests/types/test_schema.py::test_spark_schema_inference

tests/types/test_schema.py::test_spark_schema_inference

tests/types/test_schema.py::test_schema_inference_on_datatypes[data5-DataType.long]

tests/types/test_type_hints.py::test_convert_dataframe_to_example_format[data5]

tests/types/test_type_hints.py::test_convert_dataframe_to_example_format[data8]

tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

tests/utils/test_requirements_utils.py::test_gateway_extra_not_captured_when_importing_deployment_client_only

D:\a\mlflow\mlflow\mlflow\types\utils.py:452: UserWarning:

Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. If your input data contains missing values at inference time, it will be encoded as floats and will cause a schema enforcement error. The best way to avoid this problem is to infer the model schema based on a realistic data sample (training dataset) that includes missing values. Alternatively, you can declare integer columns as doubles (float64) whenever these columns may have missing values. See `Handling Integers With Missing Values <[https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values](https://www.mlflow.org/docs/latest/models.html#handling-integers-with-missing-values)>`_ for more details.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

tests/tracking/test_rest_tracking.py::test_search_traces[file]

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2648: FutureWarning:

Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2659: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2663: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2671: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_search_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2677: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

tests/tracking/test_rest_tracking.py::test_delete_traces[file]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2738: FutureWarning:

Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

tests/tracking/test_rest_tracking.py: 10 warnings

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2795: FutureWarning:

Parameter 'request_id' is deprecated and will be removed in version 3.0.0. Please use 'trace_id' instead.

tests/tracking/test_rest_tracking.py::test_link_traces_to_run_and_search_traces[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2993: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_link_traces_to_run_and_search_traces[sqlalchemy]

D:\a\mlflow\mlflow\tests\tracking\test_rest_tracking.py:2997: FutureWarning:

Parameter 'experiment_ids' is deprecated. Please use 'locations' instead.

tests/tracking/test_rest_tracking.py::test_get_logged_model_artifact[file]

tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

tests/utils/test_requirements_utils.py::test_gateway_extra_not_captured_when_importing_deployment_client_only

tests/utils/test_requirements_utils.py::test_capture_imported_modules_with_exception

tests/utils/test_requirements_utils.py::test_capture_imported_modules_extra_env_vars

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:186: UserWarning:

Add type hints to the `predict` method to enable data validation and automatic signature inference during model logging. Check [https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel](https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel) for more details.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\tests\types\test_schema.py:1784: FutureWarning:

``mlflow.models.rag_signatures.ChatCompletionRequest`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatCompletionRequest`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\mlflow\models\rag_signatures.py:25: FutureWarning:

``mlflow.models.rag_signatures.Message`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatMessage`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\tests\types\test_schema.py:1801: FutureWarning:

``mlflow.models.rag_signatures.ChatCompletionResponse`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatCompletionResponse`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\mlflow\models\rag_signatures.py:71: FutureWarning:

``mlflow.models.rag_signatures.ChainCompletionChoice`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatChoice`` instead.

tests/types/test_schema.py::test_convert_dataclass_to_schema_for_rag

D:\a\mlflow\mlflow\mlflow\models\rag_signatures.py:47: FutureWarning:

``mlflow.models.rag_signatures.Message`` is deprecated. This method will be removed in a future release. Use ``mlflow.types.llm.ChatMessage`` instead.

tests/types/test_type_hints.py::test_infer_schema_from_pydantic_model[list-expected_schema0]

D:\a\mlflow\mlflow\mlflow\types\type_hints.py:287: UserWarning:

Any type hint is inferred as AnyType, and MLflow doesn't validate the data for this type. Please use a more specific type hint to enable data validation.

tests/types/test_type_hints.py::test_infer_schema_from_python_type_hints[list-expected_schema18]

D:\a\mlflow\mlflow\mlflow\types\type_hints.py:385: UserWarning:

Any type hint is inferred as AnyType, and MLflow doesn't validate the data for this type. Please use a more specific type hint to enable data validation.

tests/utils/test_annotations.py::test_deprecated_dataclass_preserves_fields

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:270: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_annotations.py::test_deprecated_dataclass_dunder_methods_not_mutated

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:288: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_annotations.py::test_deprecated_dataclass_dunder_methods_not_mutated

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:295: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_annotations.py::test_deprecated_dataclass_dunder_methods_not_mutated

D:\a\mlflow\mlflow\tests\utils\test_annotations.py:296: FutureWarning:

``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0. This method will be removed in a future release.

tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

D:\a\mlflow\mlflow\mlflow\pyfunc\utils\data_validation.py:155: FutureWarning:

Model's `predict` method contains invalid parameters: {'inputs'}. Only the following parameter names are allowed: context, model_input, and params. Note that invalid parameters will no longer be permitted in future versions.

-- Docs: [https://docs.pytest.org/en/stable/how-to/capture-warnings.html](https://docs.pytest.org/en/stable/how-to/capture-warnings.html)

============================ slowest 10 durations =============================

30.18s setup tests/webhooks/test_e2e.py::test_model_version_created

30.02s call tests/tracking/test_client.py::test_client_get_trace_empty_result

24.67s setup tests/server/test_prometheus_exporter.py::test_metrics

24.33s setup tests/data/test_delta_dataset_source.py::test_delta_dataset_source_from_path

24.25s setup tests/tracking/test_client_webhooks.py::test_get_webhook_not_found

23.89s call tests/tracking/fluent/test_fluent.py::test_search_experiments

23.52s call tests/tracking/test_model_registry.py::test_search_registered_model_flow_paginated[file-None-<lambda>-100]

21.88s setup tests/tracking/test_rest_tracking.py::test_update_secret

20.48s call tests/utils/test_requirements_utils.py::test_capture_imported_modules_includes_gateway_extra[mlflow.gateway-True]

19.99s call tests/tracking/_tracking_service/test_utils.py::test_get_store_with_empty_mlruns

========================= per-file durations (sorted) =========================

per-file durations (sorted)

========================= command to run failed tests =========================

pytest 'tests/tracing/test_otel_logging.py::test_logging_many_traces_in_single_request'

============================== Remaining threads ==============================

1: <TMonitor(Thread-1, started daemon 6376)>

2: <TMonitor(Thread-2, started daemon 8476)>

3: <TMonitor(Thread-14, started daemon 3164)>

4: <Thread(MlflowAutologgingQueueingClient_0, started 5936)>

5: <FinalizerWorker(Thread-99, started daemon 4020)>

6: <Thread(EphemeralCacheEncryption-cleanup, started daemon 8424)>

7: <Thread(EphemeralCacheEncryption-cleanup, started daemon 2640)>

8: <Thread(EphemeralCacheEncryption-cleanup, started daemon 9412)>

9: <Thread(EphemeralCacheEncryption-cleanup, started daemon 6456)>

10: <Thread(EphemeralCacheEncryption-cleanup, started daemon 9980)>

11: <Thread(EphemeralCacheEncryption-cleanup, started daemon 6692)>

12: <Thread(EphemeralCacheEncryption-cleanup, started daemon 6668)>

13: <Thread(EphemeralCacheEncryption-cleanup, started daemon 7612)>

14: <Thread(MLflowSpanBatcherWorker, started daemon 9048)>

15: <Thread(MLflowSpanBatcherWorker, started daemon 8552)>

16: <Thread(MLflowTraceLoggingConsumer, started daemon 772)>

17: <Thread(MlflowTraceLoggingWorker_0, started daemon 8316)>

18: <Thread(OtelPeriodicExportingMetricReader, started daemon 4840)>

19: <Thread(MLflowTraceLoggingConsumer, started daemon 1252)>

20: <Thread(MlflowTraceLoggingWorker_0, started daemon 8204)>

21: <Thread(MlflowTraceLoggingWorker_1, started daemon 9844)>

22: <Thread(MlflowTraceLoggingWorker_2, started daemon 4848)>

23: <Thread(MLflowSpanBatcherWorker, started daemon 6080)>

24: <Thread(MLflowSpanBatcherWorker, started daemon 3224)>

25: <Thread(MLflowSpanBatcherWorker, started daemon 8284)>

26: <Thread(OtelBatchSpanRecordProcessor, started daemon 10196)>

27: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 8808)>

28: <Thread(MlflowLocalArtifactRepository_0, started 9224)>

29: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 5004)>

30: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 7796)>

31: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_0, started 9396)>

32: <Thread(MLflowArtifactsLoggingWorkerPool_0, started daemon 2100)>

33: <Thread(MLflowArtifactsLoggingWorkerPool_1, started daemon 1636)>

34: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_1, started 7576)>

35: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_2, started 1564)>

36: <Thread(MLflowArtifactsLoggingWorkerPool_2, started daemon 840)>

37: <Thread(MLflowArtifactsLoggingWorkerPool_3, started daemon 7300)>

38: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_3, started 5648)>

39: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_4, started 7784)>

40: <Thread(MLflowArtifactsLoggingWorkerPool_4, started daemon 7864)>

41: <Thread(MLflowAsyncArtifactsLoggingLoop, started daemon 8936)>

42: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_0, started 6716)>

43: <Thread(MLflowArtifactsLoggingWorkerPool_0, started daemon 8116)>

44: <Thread(MLflowArtifactsLoggingWorkerPool_1, started daemon 1396)>

45: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_1, started 7340)>

46: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_2, started 1732)>

47: <Thread(MLflowArtifactsLoggingWorkerPool_2, started daemon 9384)>

48: <Thread(MLflowArtifactsLoggingWorkerPool_3, started daemon 8668)>

49: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_3, started 4636)>

50: <Thread(MLflowAsyncArtifactsLoggingStatusCheck_4, started 4524)>

51: <Thread(MLflowArtifactsLoggingWorkerPool_4, started daemon 5064)>

52: <Thread(Thread-1999 (serve_forever), started daemon 8640)>

========================== Remaining child processes ==========================

1: psutil.Process(pid=8500, name='cmd.exe', status='running', started='02:05:46')

2: psutil.Process(pid=9056, name='cmd.exe', status='running', started='02:05:46')

3: psutil.Process(pid=8632, name='java.exe', status='running', started='02:05:53')

=========================== short test summary info ===========================

FAILED | MEM 4.7/16.0 GB | DISK 8.8/150.0 GB tests/tracing/test_otel_logging.py::test_logging_many_traces_in_single_request - requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=62039): Read timed out. (read timeout=10)

= 1 failed, 2204 passed, 59 skipped, 1 xfailed, 207 warnings in 1462.26s (0:24:22) =

2026/01/07 02:28:38 INFO mlflow.tracing.export.async_export_queue: Flushing the async trace logging queue before program exit. This may take a while...

2026/01/07 02:28:39 INFO mlflow.tracing.export.async_export_queue: Flushing the async trace logging queue before program exit. This may take a while...

SUCCESS: The process with PID 8632 (child process of PID 9056) has been terminated.

SUCCESS: The process with PID 9056 (child process of PID 8500) has been terminated.

SUCCESS: The process with PID 8500 (child process of PID 1376) has been terminated.

Error: Process completed with exit code 1.

1s

1s

0s

0s

0s