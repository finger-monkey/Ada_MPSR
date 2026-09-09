                                                   
                     
                    
                                
                                                                
                                 
             
                 
                                   
 
                       
                                               
                                     
                        
                        
 
                   
                            
                      
                    
                                                                         
              
                                               
                                                                    
                                        
                            
                                   
                                                           
                                                            
                    
 
                   
                                                                  
                         
                               
                                                                          
                                      
                                     
                           
                              
                                                
                                                        
                                 
                                          
                           
                      
                             
                               
                          
       
 
               
            
                                                                               
                                                                               
   
 
                            
                    
                              
                           
                                              
                                  
                                                        
                                            
                                
                               
 
                       
                
                 
 
                                        
                                      
 
             
                     
                                       
                              
                              
                                                                                 
                                             
                                                          
               
                                            
 
                                                  
 
                         
                       
                                                   
                                 
                                                               
                               
 
               
                                                                                                          
                                              
                                                
                                   
                                                              
 
                         
                                       
                                       
                                                   
                                         
 
                                                                     
          
                                       
                                                                             
                                 
                            
                                  
 
                                                                                
                                     
               
 
                  
                      
                                   
          
                                                                                                        
                            
                                       
                              
                                      
                              
       
 
                 


 
                     
                    
                                
                                                                
                                 
             
                 
                                   
 
                     
                                               
                                     
                        
                        
 
                   
                            
                      
                    
                                                                         
              
                                               
                                                                    
                                        
                            
                                   
                                                           

                        
                                                               

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sympy import preorder_traversal, Function, Symbol, pretty, sympify, lambdify
from collections import Counter
import math
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

                   
df = pd.read_csv("test_dataset_SOFC_0.8.csv")
X_geom = df[['x', 'y', 'z']].values
y_O2 = df['O2'].values
y_N2 = df['N2'].values


                 
def compute_entropy(expr):
    if expr is None:
        return 0.0
    sym_expr = expr["sympy_format"] if isinstance(expr, dict) else expr
    ops = []
    for node in preorder_traversal(sym_expr):
        if isinstance(node, Function) or isinstance(node, Symbol):
            ops.append(str(node.func))
    counter = Counter(ops)
    total = sum(counter.values())
    probs = [count / total for count in counter.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return entropy


                 
def extract_expression(pysr_result):
                       
    if pysr_result is None:
        return None, None

    try:
                            
        if hasattr(pysr_result, 'get'):
                              
            if 'sympy_format' in pysr_result:
                sympy_expr = pysr_result['sympy_format']
                equation = pysr_result['equation']
                          
            elif 'equation' in pysr_result:
                sympy_expr = pysr_result['equation']
                equation = pysr_result['equation']
            else:
                                         
                sympy_expr = str(pysr_result)
                equation = str(pysr_result)
        else:
            sympy_expr = pysr_result
            equation = str(pysr_result)

        return sympy_expr, equation
    except Exception as e:
        print(f"Expression extraction failed: {e}")
        return None, None


                    
def format_expression_display(expr_str, var_names=['x', 'y', 'z']):
                                        
    if expr_str is None:
        return "None"

           
    for i, var in enumerate(var_names):
        expr_str = expr_str.replace(f'x{i}', var)
        expr_str = expr_str.replace(f'X[{i}]', var)

    return expr_str


                    
def build_model(entropy=None, entropy_min=0.5, entropy_max=2.5):
    if entropy is None:
        entropy = entropy_max
    scale = (entropy_max - entropy) / (entropy_max - entropy_min + 1e-5)
    n_iters = int(100 + scale * 300)
    max_size = int(10 + scale * 10)
    return PySRRegressor(
        niterations=n_iters,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "sqrt"],
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        maxsize=max_size,
        verbosity=0,
        deterministic=True,
        parallelism="serial",
        random_state=42,
    )


                
fields = {
    "O2": {"target": y_O2, "expr": None, "pred": None, "loss": float("inf"),
           "sympy_expr": None, "equation_str": None, "variables": ["x", "y", "z"]},
    "N2": {"target": y_N2, "expr": None, "pred": None, "loss": float("inf"),
           "sympy_expr": None, "equation_str": None, "variables": ["x", "y", "z"]},
}

                                  
for key in fields:
    print(f"\nInitial model fitting: {key}")
    model = build_model()
    model.fit(X_geom, fields[key]["target"])
    pred = model.predict(X_geom)
    mse = np.mean((pred - fields[key]["target"]) ** 2)

             
    best_expr = model.get_best()
    sympy_expr, equation_str = extract_expression(best_expr)

    fields[key]["expr"] = best_expr
    fields[key]["sympy_expr"] = sympy_expr
    fields[key]["equation_str"] = equation_str
    fields[key]["pred"] = pred
    fields[key]["loss"] = mse

    print(f"   MSE: {mse:.6e}")
    print("   Expression:")

    if sympy_expr is not None:
        formatted_expr = format_expression_display(str(sympy_expr))
        print(f"   {formatted_expr}")

                           
        try:
            if hasattr(sympy_expr, '__sympy__'):
                sympy_obj = sympy_expr.__sympy__()
                print("\n   SymPy format:")
                print(pretty(sympy_obj, use_unicode=False))
            else:
                               
                sympy_obj = sympify(str(sympy_expr).replace('x0', 'x').replace('x1', 'y').replace('x2', 'z'))
                print("\n   SymPy format:")
                print(pretty(sympy_obj, use_unicode=False))
        except Exception as e:
            print(f"   SymPy conversion failed: {e}")
    else:
        print(f"   Expression: {str(best_expr)[:100]}...")

                     
min_rounds = 5
max_rounds = 20

for round in range(1, max_rounds + 1):
    print(f"\nAdaptive decoupling round {round}")
    print("-" * 50)

    priorities = {}
    for key, field in fields.items():
        expr = field["sympy_expr"]
        if expr is not None:
            expr_str = str(expr)
            complexity = len(expr_str)
            priorities[key] = field["loss"] * complexity
        else:
            priorities[key] = float("inf")

    to_fit = min(priorities, key=priorities.get)
    print("Variable priority scores:")
    for key, prio in priorities.items():
        print(f"   {key}: loss={fields[key]['loss']:.3e}, priority={prio:.3e}")

                       
    inputs = [X_geom]
    input_vars = ["x", "y", "z"]
    for other_key, other_field in fields.items():
        if other_key != to_fit:
            if other_field["pred"] is not None:
                inputs.append(other_field["pred"].reshape(-1, 1))
            else:
                inputs.append(other_field["target"].reshape(-1, 1))
            input_vars.append(other_key)

    X_fit = np.hstack(inputs)

            
    fields[to_fit]["variables"] = input_vars

    entropy_val = compute_entropy(fields[to_fit]["sympy_expr"]) if fields[to_fit]["sympy_expr"] is not None else 2.5
    model = build_model(entropy=entropy_val)
    model.fit(X_fit, fields[to_fit]["target"])
    y_pred = model.predict(X_fit)
    loss = np.mean((y_pred - fields[to_fit]["target"]) ** 2)

               
    best_expr = model.get_best()
    sympy_expr, equation_str = extract_expression(best_expr)

    if loss < fields[to_fit]["loss"] or round == 1:
        fields[to_fit]["loss"] = loss
        fields[to_fit]["expr"] = best_expr
        fields[to_fit]["sympy_expr"] = sympy_expr
        fields[to_fit]["equation_str"] = equation_str
        fields[to_fit]["pred"] = y_pred

    print(f"\nSelected variable: {to_fit}({', '.join(input_vars)})")
    print(f"MSE: {loss:.3e} (improvement: {fields[to_fit]['loss'] - loss:.3e})")

    if sympy_expr is not None:
                
        var_map = {}
        for i, var in enumerate(input_vars):
            var_map[f'x{i}'] = var
            var_map[f'X[{i}]'] = var

                
        expr_str = str(sympy_expr)
        for old_var, new_var in var_map.items():
            expr_str = expr_str.replace(old_var, new_var)

        print(f"Expression: {expr_str}")

                          
        try:
            sympy_obj = sympify(expr_str)
            print("\nSymPy format:")
            print(pretty(sympy_obj, use_unicode=False))
        except Exception as e:
            print(f"Warning: SymPy conversion failed: {e}")
    else:
        print("Warning: no valid expression found")

            
    if round >= min_rounds:
        all_converged = True
        for f in fields.values():
            if f["loss"] > 1e-4:
                all_converged = False
                break

        if all_converged:
            print(f"\nRound {round}: all variables converged (MSE < 1e-4); stopping.")
            break

                    
print("\n" + "=" * 80)
print("Final expressions and errors")
print("=" * 80)

            
expressions_dict = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "voltage": "0.4V",
    "num_samples": len(df),
    "fields": {}
}

for key, field in fields.items():
    print(f"\nField: {key}")
    print(f"Input variables: {', '.join(field['variables'])}")
    print(f"MSE: {field['loss']:.3e}")

    if field["sympy_expr"] is not None:
                
        var_map = {}
        for i, var in enumerate(field["variables"]):
            var_map[f'x{i}'] = var
            var_map[f'X[{i}]'] = var

                  
        expr_str = str(field["sympy_expr"])

               
        for old_var, new_var in var_map.items():
            expr_str = expr_str.replace(old_var, new_var)

        print(f"Expression string: {expr_str}")

                          
        try:
            sympy_obj = sympify(expr_str)
            print("Formatted expression:")
            print(pretty(sympy_obj, use_unicode=False))

                   
            expressions_dict["fields"][key] = {
                "expression": expr_str,
                "variables": field["variables"],
                "mse": float(field["loss"]),
                "sympy_format": str(sympy_obj),
                "variable_mapping": var_map
            }

                     
            try:
                        
                sym_vars = symbols(','.join(field["variables"]))
                      
                func = lambdify(sym_vars, sympy_obj, 'numpy')
                expressions_dict["fields"][key]["callable"] = True
                expressions_dict["fields"][key]["sympy_variables"] = [str(v) for v in sym_vars]
                print("Callable function created")
            except Exception as e:
                expressions_dict["fields"][key]["callable"] = False
                expressions_dict["fields"][key]["error"] = str(e)
                print(f"Warning: callable function creation failed: {e}")

        except Exception as e:
            print(f"Warning: SymPy conversion failed: {e}")
            expressions_dict["fields"][key] = {
                "expression": expr_str,
                "variables": field["variables"],
                "mse": float(field["loss"]),
                "error": str(e)
            }
    else:
        print("No valid solution found")
        expressions_dict["fields"][key] = {
            "expression": None,
            "variables": field["variables"],
            "mse": float(field["loss"]),
            "error": "No expression found"
        }

    print("-" * 80)


                  
def save_expressions(expressions_dict, filename=None):
                  
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SOFC_expressions_{expressions_dict['voltage']}_{timestamp}"

               
    json_file = f"{filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(expressions_dict, f, indent=2, ensure_ascii=False)
    print(f"\nExpressions saved to: {json_file}")

                        
    py_file = f"{filename}.py"
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(f"# SOFC field expressions - {expressions_dict['voltage']}\n")
        f.write(f"# Generated at: {expressions_dict['timestamp']}\n")
        f.write(f"# Samples: {expressions_dict['num_samples']}\n")
        f.write("# ==============================================================================\n\n")
        f.write("import numpy as np\nfrom sympy import symbols, sympify, lambdify\n\n")

        for field_name, field_data in expressions_dict["fields"].items():
            f.write(f"# {'=' * 60}\n")
            f.write(f"# {field_name} field\n")
            f.write(f"# MSE: {field_data['mse']:.3e}\n")
            f.write(f"# Variables: {', '.join(field_data['variables'])}\n")
            f.write(f"# {'=' * 60}\n")

            if field_data["expression"]:
                        
                vars_str = ', '.join(field_data["variables"])
                f.write(f"{field_name}_vars = symbols('{vars_str}')\n")

                       
                expr_escaped = field_data["expression"].replace("'", "\\'")
                f.write(f"{field_name}_expr = '{expr_escaped}'\n")
                f.write(f"{field_name}_sympy = sympify({field_name}_expr)\n")

                      
                f.write(f"def {field_name}_func({vars_str}):\n")
                f.write(f"    \"\"\"\n")
                f.write(f"    Evaluate the {field_name} field\n")
                f.write(f"    Expression: {field_data['expression']}\n")
                f.write(f"    MSE: {field_data['mse']:.3e}\n")
                f.write(f"    \"\"\"\n")
                f.write(f"    try:\n")
                f.write(f"        func = lambdify({field_name}_vars, {field_name}_sympy, 'numpy')\n")

                        
                params = []
                for var in field_data["variables"]:
                    params.append(var)

                f.write(f"        return func({', '.join(params)})\n")
                f.write(f"    except Exception as e:\n")
                f.write(f"        print(f\"{field_name} evaluation failed: {{e}}\")\n")
                f.write(f"        return None\n\n")

                        
                f.write(f"# Test example\n")
                f.write(f"# test_result = {field_name}_func(")
                test_params = []
                for var in field_data["variables"]:
                    if var in ['x', 'y', 'z']:
                        test_params.append(f"0.0")        
                    else:
                        test_params.append(f"0.0")           
                f.write(', '.join(test_params) + ")\n")
                f.write(f"# print(f\"{field_name}(0,0,0,...) = {{test_result}}\")\n\n")
            else:
                f.write(f"# {field_name} fitting failed\n")
                f.write(f"def {field_name}_func(*args):\n")
                f.write(f"    \"\"\"\n")
                f.write(f"    {field_name} fitting failed\n")
                f.write(f"    \"\"\"\n")
                f.write(f"    print(\"{field_name} function is unavailable\")\n")
                f.write(f"    return None\n\n")

    print(f"Python module saved to: {py_file}")

                   
    txt_file = f"{filename}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"SOFC field expressions\n")
        f.write(f"Voltage: {expressions_dict['voltage']}\n")
        f.write(f"Generated at: {expressions_dict['timestamp']}\n")
        f.write(f"Samples: {expressions_dict['num_samples']}\n")
        f.write("=" * 80 + "\n\n")

        for field_name, field_data in expressions_dict["fields"].items():
            f.write(f"{field_name}({', '.join(field_data['variables'])}):\n")
            f.write(f"MSE: {field_data['mse']:.3e}\n")
            if field_data.get("sympy_format"):
                f.write(f"Expression: {field_data['sympy_format']}\n")
            elif field_data["expression"]:
                f.write(f"Expression: {field_data['expression']}\n")
            else:
                f.write("Expression: fitting failed\n")

            if field_data.get("variable_mapping"):
                f.write(f"Variable mapping: {field_data['variable_mapping']}\n")

            f.write("-" * 80 + "\n\n")

    print(f"Text summary saved to: {txt_file}")

    return json_file, py_file, txt_file


         
save_expressions(expressions_dict)

print("\n" + "=" * 80)
print("Completed. Expressions saved in multiple formats.")
print("=" * 80)
