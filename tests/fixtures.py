"""Mixed domains for manual, real-model smoke checks."""

FINANCE_TERMS = "Central Bank, Commercial Bank, Interest Rate, Inflation, Liquidity, Asset, Liability, Equity, Bond, Stock, Derivative, Option, Credit Risk, Market Risk, Collateral, Default, Portfolio, Diversification, Capital, Yield".split(", ")
FOOD_TERMS = ["Chicken breast", "Eggs", "Hummus", "Olive oil", "Greek yogurt", "Milk", "Salmon", "Protein",
              "Poultry", "Chickpeas", "Fiber", "Dairy", "Food", "Legume"]
SOFTWARE_TERMS = ["Web Server", "API", "Database", "Cache", "Client", "Request", "Response", "Query",
                  "Authentication", "Load Balancer", "HTTP", "Application"]
LARGE_MIXED_TERMS = list(dict.fromkeys(FINANCE_TERMS + FOOD_TERMS + SOFTWARE_TERMS))
