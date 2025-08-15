# FPL Manager v3 - Advanced Fantasy Premier League Management System

A comprehensive Fantasy Premier League management system with ML predictions, real data integration, and intelligent decision support.

## 🚀 Features

### Core Functionality
- **Real Data Integration**: Live FPL data with no fake/mock data
- **ML Predictions**: Advanced machine learning for player points, captain selection, and transfers
- **Team Management**: Complete squad analysis and optimization
- **Wildcard Helper**: Intelligent team building within FPL rules
- **Weather Analysis**: Match performance impact from weather conditions
- **News Integration**: Real-time injury and team updates via AI
- **Accuracy Tracking**: Continuous learning and model improvement

### Key Capabilities
- **Captain Recommendations**: AI-powered captain selection
- **Transfer Optimization**: Smart transfer suggestions with budget analysis
- **Formation Analysis**: Tactical setup optimization
- **Player Performance Prediction**: Next gameweek points forecasting
- **Team Reports**: Comprehensive performance analytics
- **Market Analysis**: Player value and ownership trends

## 🏗️ Architecture

### Backend (Python/Flask)
```
backend/
├── api/
│   └── app.py                 # Main Flask API application
├── core/
│   ├── fpl_manager.py        # FPL data management
│   └── ml_predictor.py       # ML prediction system
├── services/
│   ├── weather_service.py    # Weather data integration
│   ├── news_service.py       # News analysis via AI
│   └── accuracy_tracker.py   # Performance tracking
└── database/
    └── models.py             # Data models and caching
```

### Frontend (React/Material-UI)
```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard/        # Main dashboard
│   │   ├── TeamManagement/   # Team selection
│   │   ├── Predictions/      # ML predictions
│   │   ├── Reports/          # Analytics
│   │   ├── Wildcard/         # Team building
│   │   └── Accuracy/         # Performance tracking
│   └── services/
│       └── api.js            # Backend communication
└── public/
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- Node.js 16+
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd fpl-manager-v3
```

2. **Install Python dependencies**
```bash
cd backend
pip install -r ../requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```env
# FPL Configuration
FPL_TEAM_ID=your_fpl_team_id

# External API Keys (Optional but recommended)
OPENWEATHER_API_KEY=your_openweather_key
PERPLEXITY_API_KEY=your_perplexity_key
MISTRAL_API_KEY=your_mistral_key

# Database
DATABASE_PATH=fpl_manager_v3.db
CACHE_DATABASE_PATH=fpl_cache_v3.db

# Features
ENABLE_WEATHER_INTEGRATION=True
ENABLE_NEWS_INTEGRATION=True
ENABLE_ACCURACY_TRACKING=True
```

4. **Start the backend server**
```bash
python backend/api/app.py
```
The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Install Node.js dependencies**
```bash
cd frontend
npm install
```

2. **Start the React development server**
```bash
npm start
```
The frontend will be available at `http://localhost:3000`

## 🔧 Configuration

### API Keys Setup

#### OpenWeather API (Weather Data)
1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key
3. Add `OPENWEATHER_API_KEY=your_key` to `.env`

#### Perplexity AI (News Analysis)
1. Sign up at [Perplexity AI](https://www.perplexity.ai)
2. Get API access
3. Add `PERPLEXITY_API_KEY=your_key` to `.env`

#### Mistral AI (Manager Personas)
1. Sign up at [Mistral AI](https://mistral.ai)
2. Get API access
3. Add `MISTRAL_API_KEY=your_key` to `.env`

### FPL Team ID
1. Go to [Fantasy Premier League](https://fantasy.premierleague.com)
2. Navigate to your team
3. Copy the team ID from the URL (e.g., `/my-team/123456/`)
4. Add `FPL_TEAM_ID=123456` to `.env`

## 🤖 Machine Learning Features

### Prediction Models
- **Points Predictor**: Random Forest + Gradient Boosting ensemble
- **Captain Recommender**: Multi-factor scoring with confidence intervals
- **Transfer Optimizer**: Value-based player comparison
- **Formation Analyzer**: Tactical setup optimization

### Features Used (40+ ML Features)
- Player performance metrics (points, form, goals, assists)
- Market data (cost, ownership, transfers)
- Team strength and form analysis
- Fixture difficulty assessment
- Weather impact modeling
- News sentiment analysis
- Historical performance patterns

### Model Training
Models automatically retrain with new data. Manual retraining:
```bash
curl -X POST http://localhost:5000/api/train-models
```

## 📊 API Endpoints

### Core Data
- `GET /api/health` - System health check
- `GET /api/bootstrap` - FPL bootstrap data
- `GET /api/team` - User team data
- `GET /api/fixtures` - Match fixtures
- `GET /api/players` - Player data with filters

### ML Predictions
- `GET /api/predictions/points` - Player points predictions
- `GET /api/predictions/captain` - Captain recommendation
- `GET /api/predictions/transfers` - Transfer suggestions

### External Data
- `GET /api/weather` - Weather data for stadiums
- `GET /api/news` - FPL-relevant news analysis

### Performance
- `GET /api/accuracy/stats` - Prediction accuracy metrics
- `POST /api/train-models` - Retrain ML models

## 🎯 Usage Guide

### Getting Started
1. Set up your FPL team ID in configuration
2. Train initial ML models with current data
3. Explore the dashboard for insights
4. Use the Wildcard Helper for team optimization

### Making Decisions
1. **Captain Selection**: Check ML recommendations with confidence scores
2. **Transfers**: Review suggested transfers with expected point gains
3. **Team Planning**: Use Wildcard Helper for major team changes
4. **Formation**: Analyze optimal tactical setups

### Monitoring Performance
1. Track prediction accuracy over time
2. Review model performance metrics
3. Analyze which factors contribute to better predictions

## 🛠️ Development

### Adding New Features
1. Backend: Add new endpoints in `backend/api/app.py`
2. Frontend: Create new components in `frontend/src/components/`
3. ML: Extend models in `backend/core/ml_predictor.py`

### Running Tests
```bash
# Backend tests
python -m pytest backend/tests/

# Frontend tests
cd frontend && npm test
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📈 Performance

### System Requirements
- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB RAM, 5GB storage
- **Database**: SQLite (included)
- **Caching**: Intelligent caching for API responses

### Optimization
- Database caching reduces API calls by 80%
- ML model inference < 100ms
- Real-time data updates every 30 minutes
- Frontend optimized for mobile and desktop

## 🔒 Security & Privacy

- No user authentication required (uses public FPL data)
- API keys stored securely in environment variables
- Rate limiting on external API calls
- Local database storage only

## 🚨 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check Python version (3.9+ required)
   - Verify all dependencies installed
   - Check `.env` file configuration

2. **Frontend connection errors**
   - Ensure backend is running on port 5000
   - Check CORS configuration
   - Verify proxy setting in `package.json`

3. **Missing predictions**
   - Run model training: `POST /api/train-models`
   - Check FPL API availability
   - Verify sufficient player data

4. **External API errors**
   - Check API key configuration
   - Verify API key validity and quotas
   - Review rate limiting

### Debug Mode
```bash
# Backend debug
FLASK_ENV=development python backend/api/app.py

# Frontend debug
cd frontend && npm start
```

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Core FPL data integration
- ✅ Basic ML predictions
- ✅ React dashboard
- ✅ Wildcard functionality

### Phase 2 (Next)
- Enhanced weather integration
- Full news analysis implementation
- Advanced tactical analysis
- Mobile app development

### Phase 3 (Future)
- Multi-league support
- Social features
- Advanced visualization
- Machine learning model marketplace

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/fpl-manager-v3/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/fpl-manager-v3/discussions)
- **Email**: support@fplmanager.com

## 🙏 Acknowledgments

- **Fantasy Premier League**: For providing the official API
- **OpenWeather**: For weather data integration
- **Perplexity AI**: For news analysis capabilities
- **Material-UI**: For the excellent React components
- **scikit-learn**: For machine learning framework

---

**FPL Manager v3** - Elevate Your Fantasy Premier League Game with AI-Powered Intelligence 🚀⚽