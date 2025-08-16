# Mistral AI Integration Setup

## Overview
The FPL Manager v3 now includes advanced team optimization using **Mistral AI** with a **blackboard technique**. This provides sophisticated multi-agent analysis for FPL team selection.

## Features

### 🧠 **Blackboard Architecture**
- **Formation Specialist**: Analyzes tactical formations and position balance
- **Value Analyzer**: Identifies best value players and budget allocation
- **Risk Assessor**: Evaluates injury risk, rotation risk, and differential picks
- **Fixture Analyzer**: Considers upcoming fixtures and difficulty
- **Form Tracker**: Analyzes player form and momentum
- **Master Optimizer**: Synthesizes all agent insights into final strategy

### 🎯 **Advanced Analysis**
- **Multi-agent reasoning** with collaborative decision making
- **Risk tolerance settings**: Conservative, Balanced, Aggressive
- **Strategic insights** beyond basic point predictions
- **Formation-specific recommendations**
- **Budget allocation strategies**

## Setup Instructions

### 1. Get Mistral API Key
1. Sign up at [console.mistral.ai](https://console.mistral.ai)
2. Create a new API key
3. Copy the key (starts with `mistral-`)

### 2. Configure Environment
Add your Mistral API key to your environment:

**Windows (Command Prompt):**
```cmd
set MISTRAL_API_KEY=your-mistral-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:MISTRAL_API_KEY="your-mistral-api-key-here"
```

**Linux/Mac:**
```bash
export MISTRAL_API_KEY=your-mistral-api-key-here
```

### 3. Restart Backend
Restart the backend server to pick up the environment variable:
```bash
venv/Scripts/python.exe -m backend.api.app
```

## Usage

### Frontend Interface
1. Navigate to the **Wildcard Helper** page
2. Toggle **"🧠 Use Mistral AI (Advanced)"** switch
3. Configure your preferences:
   - **Budget**: Set your available budget (£80M - £120M)
   - **Formation**: Choose your preferred formation
   - **Risk Tolerance**: 
     - 🛡️ **Conservative**: Safe picks, proven players
     - ⚖️ **Balanced**: Mix of safety and upside
     - 🚀 **Aggressive**: High-risk/high-reward differentials

4. Click **"🧠 Analyze with Mistral AI"**

### API Endpoint
```http
POST /api/wildcard/optimize-mistral
Content-Type: application/json

{
  "budget": 100.0,
  "formation": "3-4-3",
  "risk_tolerance": "balanced",
  "constraints": {}
}
```

## Response Format

```json
{
  "success": true,
  "data": {
    "mistral_strategy": {
      "overall_strategy_summary": "Strategic recommendations...",
      "position_priorities": {...},
      "budget_allocation": {...},
      "key_insights": [...]
    },
    "blackboard_summary": {
      "total_entries": 5,
      "agents": ["formation_specialist", "value_analyzer", "risk_assessor"],
      "latest_entries": [...]
    },
    "agents_consulted": ["formation_specialist", "value_analyzer"],
    "constraints": {...}
  }
}
```

## Benefits Over Standard ML

| Feature | Standard ML | Mistral AI |
|---------|-------------|------------|
| **Analysis Depth** | Points prediction only | Multi-dimensional strategy |
| **Risk Management** | Basic | Advanced risk profiling |
| **Tactical Insight** | Formation compliance | Formation optimization |
| **Budget Strategy** | Spend efficiently | Strategic allocation plans |
| **Decision Making** | Single algorithm | Collaborative agents |
| **Adaptability** | Fixed strategies | Dynamic reasoning |

## Troubleshooting

### "Mistral AI not configured" Error
- Ensure `MISTRAL_API_KEY` environment variable is set
- Restart the backend server after setting the variable
- Check that your API key is valid and active

### No Strategy Results
- Verify your Mistral account has API credits
- Check network connectivity
- Review backend logs for API call errors

### Partial Results
- Some agents may fail while others succeed
- The system will synthesize available insights
- Check the `blackboard_summary` to see which agents contributed

## Cost Considerations
- Each optimization uses multiple AI calls (3-5 requests)
- Mistral charges per token used
- Typical optimization costs $0.05-$0.15
- Consider using for important decisions (wildcards, major transfers)

## Fallback Behavior
If Mistral AI is unavailable:
- Standard ML optimization remains fully functional
- Toggle off "Use Mistral AI" to use regular optimization
- No impact on core FPL Manager functionality

---

**Note**: Mistral AI provides strategic analysis and insights. For actual team selection, combine these insights with the regular ML optimization or manual player selection.