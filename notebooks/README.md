# AI On-Call Agent - ML Training Notebook

This notebook demonstrates the machine learning capabilities of the AI On-Call Agent system, including incident classification, action recommendation, and performance analysis.

## Features Demonstrated

- ✅ **ML Service Initialization** - Setup and configuration of ML components
- ✅ **Incident Severity Prediction** - Classify incidents by severity (low/medium/high/critical)
- ✅ **Action Recommendations** - Suggest appropriate remediation actions
- ✅ **Model Training** - Train ML models with sample data
- ✅ **Performance Analysis** - Measure prediction speed and accuracy
- ✅ **Data Visualization** - Charts and graphs showing prediction distributions
- ✅ **Real-time Testing** - Test predictions on new incident scenarios

## Setup Instructions

### 1. Install Dependencies

```bash
# Navigate to the notebooks directory
cd notebooks

# Install required packages
pip install -r requirements.txt
```

### 2. Alternative Installation

If you prefer to install packages individually:

```bash
pip install matplotlib seaborn pandas numpy scikit-learn pydantic-settings python-dotenv structlog rich sqlalchemy asyncpg greenlet jupyter
```

### 3. Start Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook ml_training_complete.ipynb

# Or use Jupyter Lab
jupyter lab ml_training_complete.ipynb
```

## Notebook Structure

1. **Setup & Package Verification** - Check all required packages are installed
2. **Import Libraries** - Load all necessary Python libraries
3. **ML Service Initialization** - Setup the ML service in demo mode
4. **Incident Severity Testing** - Test classification on sample incidents
5. **Action Recommendation Testing** - Test action suggestions
6. **Data Visualization** - Create charts showing prediction distributions
7. **Model Training** - Train new models with sample data
8. **Model Evaluation** - Assess model performance and accuracy
9. **AI Engine Integration** - Test integration with the AI decision engine
10. **Performance Analysis** - Measure prediction speed and create performance charts
11. **Final Demo** - Test on new scenarios and show system status

## Demo Mode Features

This notebook runs in **demo mode** which means:

- ✅ Works without database connectivity
- ✅ Uses in-memory ML models
- ✅ Generates sample training data
- ✅ Provides fallback predictions
- ✅ Shows all core ML functionality

## Expected Results

When you run the notebook successfully, you should see:

- **10 incident scenarios** with severity predictions
- **10 action recommendations** for the same scenarios
- **4 visualization charts** showing data distributions
- **Model training results** with ~85% accuracy
- **Performance metrics** showing prediction speeds
- **5 new test scenarios** with real-time predictions

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all packages from `requirements.txt` are installed
2. **Database Errors**: Normal in demo mode - the notebook handles these gracefully
3. **Sklearn Warnings**: These are normal and don't affect functionality
4. **Performance Variations**: Prediction times may vary based on your system

### Package Issues

If you encounter package installation issues:

```bash
# Update pip first
pip install --upgrade pip

# Install packages one by one
pip install matplotlib
pip install seaborn
pip install pandas
# ... etc
```

### Jupyter Issues

If Jupyter doesn't start:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Or try with conda
conda install jupyter
```

## Production Usage

In a production environment, this ML system would:

- Connect to a real PostgreSQL database
- Train on actual incident and resolution data
- Integrate with monitoring systems (ELK, Splunk, etc.)
- Provide real-time incident analysis
- Support continuous model retraining

## Support

For issues or questions about this notebook:

1. Check the package requirements are met
2. Review the troubleshooting section above
3. Ensure you're running Python 3.8+ with async support
4. Verify Jupyter is properly installed and configured

## License

This notebook is part of the AI On-Call Agent project and follows the same licensing terms.
