print("=== Starting Server ===")
print("Importing app...")
try:
    from test_server import app
    print("App imported successfully")
    
    print("\n=== App Configuration ===")
    print(f"Name: {app.name}")
    print(f"Debug: {app.debug}")
    print(f"URL Map: {app.url_map}")
    
    print("\n=== Starting Server on http://0.0.0.0:5006 ===")
    app.run(host='0.0.0.0', port=5006, debug=True, use_reloader=False)
except Exception as e:
    print("\n=== ERROR ===")
    print(f"Failed to start server: {str(e)}")
    print("\n=== Stack Trace ===")
    import traceback
    traceback.print_exc()
