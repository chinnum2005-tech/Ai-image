import React from 'react';
import { Shield, Github, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

const Header = () => {
  return (
    <header className="glass-effect border-b border-white/10">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-3"
          >
            <div className="gradient-border rounded-lg">
              <div className="bg-slate-900 p-2 rounded-md">
                <Shield className="w-8 h-8 text-blue-400" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">AI Detector</h1>
              <p className="text-xs text-gray-400">Pixel-wise Analysis</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-4"
          >
            <div className="flex items-center space-x-2 text-green-400">
              <Activity className="w-5 h-5" />
              <span className="text-sm">System Active</span>
            </div>
            
            <button className="text-gray-400 hover:text-white transition-colors">
              <Github className="w-6 h-6" />
            </button>
          </motion.div>
        </div>
      </div>
    </header>
  );
};

export default Header;
