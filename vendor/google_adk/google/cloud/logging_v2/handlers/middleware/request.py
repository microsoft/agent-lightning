# Copyright 2017 Google LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Django middleware helper to capture a request.

The request is stored on a thread-local so that it can be
inspected by other helpers.
"""

import threading


_thread_locals = threading.local()


def _get_django_request():
    """Get Django request from thread local.

    Returns:
        str: Django request
    """
    return getattr(_thread_locals, "request", None)


def RequestMiddleware(get_response):
    """Saves the request in thread local"""

    def middleware(request):
        """Called on each request, before Django decides which view to execute.

        Args:
            request(django.http.request.HttpRequest):
                Django http request.
        """
        _thread_locals.request = request
        if get_response:
            return get_response(request)
        else:
            return None

    return middleware
