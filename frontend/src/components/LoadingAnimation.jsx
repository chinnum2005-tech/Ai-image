import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Eye, Cpu, Zap } from 'lucide-react';

const LoadingAnimation = () => {
  const icons = [Brain, Eye, Cpu, Zap];
  
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="relative w-32 h-32 mb-8">
        {/* Rotating circle */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          className="absolute inset-0 rounded-full border-4 border-purple-500/20 border-t-purple-500"
        />
        
        {/* Inner circle */}
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="absolute inset-4 rounded-full border-4 border-blue-500/20 border-r-blue-500"
        />
        
        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="p-4 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full"
          >
            <Brain className="w-8 h-8 text-white" />
          </motion.div>
        </div>
      </div>
      
      {/* Loading text */}
      <motion.h3
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="text-xl font-semibold text-white mb-2"
      >
        Analyzing Image
      </motion.h3>
      
      {/* Progress stages */}
      <div className="space-y-2 w-64">
        {[
          'Extracting pixel features...',
          'Analyzing PRNU patterns...',
          'Computing ELA map...',
          'Running deep learning model...'
        ].map((stage, index) => (
          <motion.div
            key={stage}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.2 }}
            className="flex items-center space-x-2"
          >
            <motion.div
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 2, repeat: Infinity, delay: index * 0.3 }}
            >
              {React.createElement(icons[index], {
                className: "w-4 h-4 text-purple-400"
              })}
            </motion.div>
            <motion.span
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity, delay: index * 0.3 }}
              className="text-sm text-gray-400"
            >
              {stage}
            </motion.span>
          </motion.div>
        ))}
      </div>
      
      {/* Progress bar */}
      <div className="w-64 h-2 bg-gray-800 rounded-full mt-6 overflow-hidden">
        <motion.div
          animate={{ x: ["-100%", "200%"] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="h-full w-1/3 bg-gradient-to-r from-blue-500 to-purple-500"
        />
      </div>
    </div>
  );
};

export default LoadingAnimation;
