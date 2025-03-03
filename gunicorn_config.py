bind = "0.0.0.0:10000"
workers = 1  # Reduce from 2 to 1
threads = 2  # Use threads instead of multiple workers
timeout = 120
max_requests = 5
max_requests_jitter = 2