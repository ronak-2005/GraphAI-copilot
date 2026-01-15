const DataViz = () => {
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* Floating Chart Elements */}
      <svg
        viewBox="0 0 400 400"
        className="w-full max-w-md h-auto"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Background Circles */}
        <circle cx="200" cy="200" r="150" fill="white" fillOpacity="0.05" />
        <circle cx="200" cy="200" r="100" fill="white" fillOpacity="0.03" />

        {/* Bar Chart Group */}
        <g className="animate-float">
          <rect x="60" y="180" width="30" height="80" rx="4" fill="hsl(187 85% 43%)" className="animate-bar-grow animation-delay-100" />
          <rect x="100" y="140" width="30" height="120" rx="4" fill="hsl(25 95% 53%)" className="animate-bar-grow animation-delay-200" />
          <rect x="140" y="160" width="30" height="100" rx="4" fill="hsl(173 80% 40%)" className="animate-bar-grow animation-delay-300" />
          <rect x="180" y="120" width="30" height="140" rx="4" fill="hsl(258 90% 66%)" className="animate-bar-grow animation-delay-400" />
          <rect x="220" y="150" width="30" height="110" rx="4" fill="hsl(187 85% 43%)" className="animate-bar-grow animation-delay-500" />
        </g>

        {/* Line Chart */}
        <g className="animate-float-delay">
          <path
            d="M280 250 L300 220 L320 240 L340 180 L360 200"
            stroke="hsl(25 95% 53%)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
            className="animate-line-draw"
          />
          <circle cx="280" cy="250" r="5" fill="hsl(25 95% 53%)" className="animate-pulse-glow" />
          <circle cx="300" cy="220" r="5" fill="hsl(25 95% 53%)" className="animate-pulse-glow" />
          <circle cx="320" cy="240" r="5" fill="hsl(25 95% 53%)" className="animate-pulse-glow" />
          <circle cx="340" cy="180" r="5" fill="hsl(25 95% 53%)" className="animate-pulse-glow" />
          <circle cx="360" cy="200" r="5" fill="hsl(25 95% 53%)" className="animate-pulse-glow" />
        </g>

        {/* Pie Chart */}
        <g className="animate-float-slow" transform="translate(300, 100)">
          <circle cx="0" cy="0" r="35" fill="hsl(217 91% 30%)" />
          <path d="M0 0 L0 -35 A35 35 0 0 1 30 17 Z" fill="hsl(187 85% 43%)" />
          <path d="M0 0 L30 17 A35 35 0 0 1 -20 28 Z" fill="hsl(25 95% 53%)" />
          <circle cx="0" cy="0" r="15" fill="hsl(217 91% 20%)" />
        </g>

        {/* Floating Data Points */}
        <g>
          <circle cx="80" cy="100" r="8" fill="hsl(187 85% 43%)" fillOpacity="0.8" className="animate-pulse-glow" />
          <circle cx="320" cy="320" r="6" fill="hsl(25 95% 53%)" fillOpacity="0.8" className="animate-pulse-glow animation-delay-200" />
          <circle cx="350" cy="280" r="4" fill="hsl(173 80% 40%)" fillOpacity="0.8" className="animate-pulse-glow animation-delay-400" />
        </g>

        {/* Dashboard Frame */}
        <g className="animate-float-delay" transform="translate(50, 280)">
          <rect x="0" y="0" width="120" height="80" rx="8" fill="white" fillOpacity="0.1" stroke="white" strokeOpacity="0.2" strokeWidth="1" />
          <rect x="10" y="10" width="100" height="4" rx="2" fill="white" fillOpacity="0.3" />
          <rect x="10" y="20" width="60" height="3" rx="1.5" fill="white" fillOpacity="0.2" />
          <rect x="10" y="35" width="40" height="35" rx="4" fill="hsl(187 85% 43%)" fillOpacity="0.6" />
          <rect x="55" y="35" width="55" height="35" rx="4" fill="hsl(25 95% 53%)" fillOpacity="0.6" />
        </g>

        {/* Gauge Chart */}
        <g transform="translate(100, 80)" className="animate-float">
          <path
            d="M-30 0 A30 30 0 0 1 30 0"
            stroke="white"
            strokeOpacity="0.2"
            strokeWidth="8"
            strokeLinecap="round"
            fill="none"
          />
          <path
            d="M-30 0 A30 30 0 0 1 15 -26"
            stroke="hsl(173 80% 40%)"
            strokeWidth="8"
            strokeLinecap="round"
            fill="none"
          />
          <circle cx="0" cy="0" r="6" fill="white" fillOpacity="0.9" />
        </g>
      </svg>

      {/* Floating Labels */}
      <div className="absolute top-8 right-8 bg-card/10 backdrop-blur-sm rounded-lg px-4 py-2 border border-primary-foreground/10 animate-float">
        <span className="text-primary-foreground/90 text-sm font-medium">+24.5%</span>
      </div>
      <div className="absolute bottom-16 left-8 bg-card/10 backdrop-blur-sm rounded-lg px-4 py-2 border border-primary-foreground/10 animate-float-delay">
        <span className="text-primary-foreground/90 text-sm font-medium">Live Data</span>
      </div>
    </div>
  );
};

export default DataViz;
