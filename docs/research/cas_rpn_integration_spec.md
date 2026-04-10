# Sovereign CAS Design for K3D RPN Model

## 1. New RPN Opcodes for CAS (0x1A0-0x1BF Range)

| Hex | Decimal | Name | Description | Stack Behavior |
|-----|---------|------|-------------|----------------|
| 0x1A0 | 416 | `OP_POLY_COEFF` | Define polynomial coefficient | pushes coeff to poly stack |
| 0x1A1 | 417 | `OP_POLY_ADD` | Add two polynomials | poly1, poly2 → poly_sum |
| 0x1A2 | 418 | `OP_POLY_MUL` | Multiply polynomials | poly1, poly2 → poly_product |
| 0x1A3 | 419 | `OP_POLY_DIV` | Polynomial division (quotient) | num, den → quotient |
| 0x1A4 | 420 | `OP_POLY_REM` | Polynomial remainder | num, den → remainder |
| 0x1A5 | 421 | `OP_POLY_GCD` | Greatest common divisor | poly1, poly2 → gcd |
| 0x1A6 | 422 | `OP_POLY_EXPAND` | Expand polynomial product | poly → expanded |
| 0x1A7 | 423 | `OP_POLY_FACTOR` | Factor polynomial | poly → factor_list |
| 0x1A8 | 424 | `OP_SIMPLIFY` | Algebraic simplification | expr → simplified |
| 0x1A9 | 425 | `OP_SUBSTITUTE` | Variable substitution | expr, var, val → substituted |
| 0x1AA | 426 | `OP_EQUATION` | Create equation | lhs, rhs → equation |
| 0x1AB | 427 | `OP_SOLVE` | Solve equation | equation, var → solutions |
| 0x1AC | 428 | `OP_COLLECT` | Collect terms | expr, var → collected |
| 0x1AD | 429 | `OP_EXPAND_POW` | Expand power | base, exp → expanded |
| 0x1AE | 430 | `OP_RATIONALIZE` | Rationalize expression | expr → rational |
| 0x1AF | 431 | `OP_TRIG_SIMPLIFY` | Trigonometric simplification | expr → simplified |
| 0x1B0 | 432 | `OP_LOG_SIMPLIFY` | Logarithmic simplification | expr → simplified |
| 0x1B1 | 433 | `OP_PATTERN_MATCH` | Pattern matching | expr, pattern → matches |
| 0x1B2 | 434 | `OP_RULE_APPLY` | Apply transformation rule | expr, rule → transformed |
| 0x1B3 | 435 | `OP_EVAL_SYM` | Symbolic evaluation | expr → evaluated |
| 0x1B4 | 436 | `OP_COEFF_EXTRACT` | Extract coefficient | poly, var, power → coeff |

## 2. Symbolic Expression Encoding in RPN

### Expression Representation
Symbolic expressions are encoded as RPN programs where:
- Variables are `OP_VAR_X`, `OP_VAR_Y`, etc.
- Constants are `OP_CONST` + value
- Operators are standard math opcodes
- Polynomials use coefficient sequences with `OP_POLY_COEFF`

### Example: "d/dx(x² + sin(x))" as RPN
```
OP_VAR_X          # Push variable x
OP_CONST, 2       # Push constant 2
OP_POWER          # x²
OP_VAR_X          # Push x
OP_SIN            # sin(x)
OP_ADD            # x² + sin(x)
OP_SYMBOLIC_DIFF  # d/dx(...)
```

### Polynomial Encoding
Polynomial in x: `3x³ + 2x + 5`
```
OP_CONST, 3       # coefficient for x³
OP_POLY_COEFF
OP_CONST, 0       # coefficient for x² (zero)
OP_POLY_COEFF
OP_CONST, 2       # coefficient for x
OP_POLY_COEFF
OP_CONST, 5       # constant term
OP_POLY_COEFF
OP_VAR_X          # variable
OP_POLY_BUILD     # builds polynomial from stack
```

## 3. Minimal CAS Kernel Set (PTX Functions)

### Core PTX Functions:
1. **Polynomial Arithmetic** (64-bit floating point)
   - `poly_add(double*, double*, int, int)` - Add polynomials
   - `poly_mul(double*, double*, int, int)` - Multiply polynomials
   - `poly_div(double*, double*, int, int, double**, double**)` - Division with remainder

2. **Symbolic Differentiation Rules**
   - `diff_power(double base, double exp, double dbase, double dexp)`
   - `diff_composite(uint32_t opcode, double inner, double dinner)`
   - `diff_trig(uint32_t opcode, double arg, double darg)`

3. **Simplification Core**
   - `simplify_rational(double num[], double den[], int n, int d)`
   - `collect_terms(double coeffs[], int max_degree)`
   - `gcd_poly(double* p1, double* p2, int n1, int n2)`

4. **Equation Solving** (quadratic/cubic)
   - `solve_quadratic(double a, double b, double c, double roots[2])`
   - `solve_cubic(double a, double b, double c, double d, double roots[3])`

### Minimal Complete Set:
- Polynomial arithmetic (all operations)
- Basic differentiation rules (power, sum, product, chain rule)
- Rational simplification
- Linear equation solving
- Pattern matching engine for rule application

## 4. Connection to Math Galaxy

### Star Entries for CAS:
```
StarEntry {
    uuid: "poly-coeff-3x^2+2x+1",
    meaning_rpn: [OP_POLY_COEFF, 3, OP_POLY_COEFF, 2, OP_POLY_COEFF, 1],
    visual_rpn: [OP_TEXT, "3x²+2x+1"],
    behavior_rpn: [OP_POLY_EXPAND, OP_SIMPLIFY],
    metadata: {
        type: "polynomial",
        degree: 2,
        variable: "x",
        coefficients: [3, 2, 1]
    }
}
```

### Grammar Galaxy Entries (Transformation Rules):
```
GrammarRule {
    pattern: [OP_VAR_X, OP_CONST, 2, OP_POWER, OP_SYMBOLIC_DIFF],
    replacement: [OP_VAR_X, OP_CONST, 2, OP_MUL],
    condition: "differentiating x^n"
}
```

### Coefficient Storage:
- Polynomial coefficients become `StarEntry` with type "coefficient-sequence"
- Each CAS transformation rule is a `GrammarEntry`
- Common expressions (like sin²x + cos²x = 1) become `MeaningCentricStar` entries

## 5. PTX Kernel Sketch for OP_SYMBOLIC_DIFF

```ptx
// Kernel: symbolic_diff
// Input: RPN program in global memory, variable index to differentiate wrt
// Output: Derivative RPN program in output buffer

.func .reg .b32 diff_kernel(
    .param .b64 rpn_program,
    .param .b32 program_len,
    .param .b32 var_index,
    .param .b64 output_buf
) {
    .reg .b32 stack[128];
    .reg .b32 sp;
    .reg .b32 opcode;
    .reg .f64 val1, val2, result;
    .reg .pred p1, p2;
    
    // Initialize
    mov.b32 sp, 0;
    
diff_loop:
    // Read opcode from RPN program
    ld.global.u32 opcode, [rpn_program];
    add.u64 rpn_program, rpn_program, 4;
    
    // Dispatch based on opcode
    .reg .b32 case_var, case_const, case_add, case_mul, case_power;
    setp.eq.u32 p1, opcode, OP_VAR_X;
    setp