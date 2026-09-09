











from sympy import Add,Mul
import regex


import re

def convert_pow_to_div(expr_str):

    pattern = r"Pow\((.*),-1\)"

    new_expr_str = re.sub(pattern, r"div(1,\1)", expr_str)
    return new_expr_str






def convert_power(string):

    try:
        if "**" in string:

            import re
            matches = re.finditer(r"\w+\*\*\d+", string)

            new_powers = []

            for match in matches:

                power = match.group()

                base, exponent = power.split("**")

                exponent = int(exponent)

                new_power = "*".join([base] * exponent)

                new_powers.append(new_power)

            new_string = re.sub(r"\w+\*\*\d+", lambda m: new_powers.pop(0), string)

            return new_string

        else:
            return string
    except:
        return str(string)

def convert_power_sin(string):

    pattern = r"(sin|log)\((?:[^()]+|(?R))*\)\*{2}\d+"
    while "**" in string:

        matches = regex.finditer(pattern,string)

        new_powers = []

        for match in matches:

            power = match.group()

            base, exponent = power.rsplit("**",1)

            exponent = int(exponent)

            new_power = "*".join([base] * exponent)

            new_powers.append(new_power)

        string = regex.sub(pattern, lambda m: new_powers.pop(0), string)


    return string



def to_prefix(expr):
    if expr.is_Atom:
        return str(expr)
    else:
        op = expr.func.__name__
        if isinstance(expr, Add):
            if len(expr.args) == 1:
                return to_prefix(expr.args[0])
            else:
                return f'{op}({to_prefix(expr.args[-1])}, {to_prefix(Add(*expr.args[:-1]))})'
        if isinstance(expr, Mul):
            if len(expr.args) == 1:
                return to_prefix(expr.args[0])
            else:
                return f'{op}({to_prefix(expr.args[-1])}, {to_prefix(Mul(*expr.args[:-1],evaluate=False))})'
        else:
            args = ','.join(to_prefix(arg) for arg in expr.args)
            return f'{op}({args})'
