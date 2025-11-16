import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Layers, Activity, ChevronLeft, ChevronRight } from 'lucide-react';

const FeatureVisualization = ({ visualizations }) => {
  const [activeView, setActiveView] = useState('prnu');
  
  const views = [
    {
      id: 'prnu',
      name: 'PRNU Analysis',
      icon: Activity,
      description: 'Photo Response Non-Uniformity reveals sensor noise patterns',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      id: 'ela',
      name: 'Error Level Analysis',
      icon: Layers,
      description: 'Compression artifact analysis highlights manipulated regions',
      color: 'from-purple-500 to-pink-500'
    }
  ];

  const currentView = views.find(v => v.id === activeView);
  const ViewIcon = currentView?.icon || Eye;

  const handlePrevious = () => {
    const currentIndex = views.findIndex(v => v.id === activeView);
    const prevIndex = (currentIndex - 1 + views.length) % views.length;
    setActiveView(views[prevIndex].id);
  };

  const handleNext = () => {
    const currentIndex = views.findIndex(v => v.id === activeView);
    const nextIndex = (currentIndex + 1) % views.length;
    setActiveView(views[nextIndex].id);
  };

  if (!visualizations || (!visualizations.prnu && !visualizations.ela)) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="glass-effect rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white flex items-center">
          <Eye className="w-5 h-5 mr-2" />
          Feature Visualizations
        </h3>
        
        {/* View Switcher */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={handlePrevious}
            className="p-2 glass-effect rounded-lg text-white hover:bg-white/20 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </motion.button>
          
          <div className="flex space-x-1">
            {views.map((view) => (
              <button
                key={view.id}
                onClick={() => setActiveView(view.id)}
                className={`w-2 h-2 rounded-full transition-all ${
                  activeView === view.id
                    ? 'w-8 bg-gradient-to-r ' + view.color
                    : 'bg-white/30'
                }`}
              />
            ))}
          </div>
          
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={handleNext}
            className="p-2 glass-effect rounded-lg text-white hover:bg-white/20 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </motion.button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeView}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="space-y-4"
        >
          {/* View Header */}
          <div className="flex items-center space-x-3 mb-4">
            <div className={`p-3 rounded-lg bg-gradient-to-br ${currentView.color}`}>
              <ViewIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h4 className="text-white font-semibold">{currentView.name}</h4>
              <p className="text-sm text-gray-400">{currentView.description}</p>
            </div>
          </div>

          {/* Feature Map Display */}
          <div className="relative rounded-lg overflow-hidden shadow-2xl">
            {visualizations[activeView] ? (
              <motion.img
                initial={{ scale: 0.95 }}
                animate={{ scale: 1 }}
                src={visualizations[activeView]}
                alt={currentView.name}
                className="w-full h-auto"
              />
            ) : (
              <div className="h-64 flex items-center justify-center bg-gray-800/50">
                <p className="text-gray-400">Visualization not available</p>
              </div>
            )}
            
            {/* Overlay gradient for effect */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />
          </div>

          {/* Analysis Legend */}
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="glass-effect rounded-lg p-3">
              <h5 className="text-xs font-semibold text-blue-400 mb-1">High Intensity</h5>
              <p className="text-xs text-gray-400">
                Red/yellow regions indicate strong feature presence
              </p>
            </div>
            <div className="glass-effect rounded-lg p-3">
              <h5 className="text-xs font-semibold text-purple-400 mb-1">Low Intensity</h5>
              <p className="text-xs text-gray-400">
                Blue/dark regions show minimal feature detection
              </p>
            </div>
          </div>

          {/* Additional Info */}
          <div className="glass-effect rounded-lg p-4">
            <div className="flex items-start space-x-2">
              <Info className="w-4 h-4 text-yellow-400 mt-0.5" />
              <div className="text-xs text-gray-300">
                {activeView === 'prnu' ? (
                  <p>
                    PRNU analysis reveals camera sensor noise patterns. Authentic photos typically show
                    consistent noise patterns unique to the capturing device, while AI-generated images
                    lack these natural sensor characteristics.
                  </p>
                ) : (
                  <p>
                    ELA highlights differences in compression levels across the image. Uniform compression
                    suggests authentic capture, while varying levels may indicate AI generation or editing.
                  </p>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
};

// Missing import
import { Info } from 'lucide-react';

export default FeatureVisualization;
