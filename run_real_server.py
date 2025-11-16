import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the app module directly
import app

# Run the app
if __name__ == "__main__":
    app.app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)