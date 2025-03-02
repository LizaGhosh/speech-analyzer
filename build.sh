#!/bin/bash
pip install -r requirements.txt
chmod -R 755 static
mkdir -p /tmp/static_cache
cp -r static/* /tmp/static_cache/