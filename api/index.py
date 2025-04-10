from flask import Flask, Response, request
import app

def handler(request, context):
    """Handle requests in the Vercel Functions format."""
    return app.app(request)