#pragma once

#include <math.h>
#include <stdint.h>

#define RPN_OP_PUSH_OPERAND_0 0x10u
#define RPN_OP_PUSH_OPERAND_1 0x11u
#define RPN_OP_ADD 0x20u
#define RPN_OP_SUB 0x21u
#define RPN_OP_MUL 0x22u
#define RPN_OP_DIV 0x23u
#define RPN_OP_POW 0x24u
#define RPN_OP_STORE_RESULT 0x30u
#define RPN_OP_RET 0xFFu

#define RPN_STACK_DEPTH 16

__device__ __forceinline__ int rpn_execute_device(
    const unsigned char* __restrict__ program_table,
    unsigned int program_offset,
    unsigned int program_length,
    int operand_0,
    int operand_1,
    int* result_out
) {
    if (program_table == nullptr || result_out == nullptr || program_length == 0u) {
        return 0;
    }

    float stack[RPN_STACK_DEPTH];
    int stack_pointer = 0;
    float result = 0.0f;

    for (unsigned int pc = 0u; pc < program_length; ++pc) {
        const unsigned char opcode = program_table[program_offset + pc];
        switch (opcode) {
            case RPN_OP_PUSH_OPERAND_0:
                if (stack_pointer >= RPN_STACK_DEPTH) {
                    return 0;
                }
                stack[stack_pointer++] = static_cast<float>(operand_0);
                break;
            case RPN_OP_PUSH_OPERAND_1:
                if (stack_pointer >= RPN_STACK_DEPTH) {
                    return 0;
                }
                stack[stack_pointer++] = static_cast<float>(operand_1);
                break;
            case RPN_OP_ADD:
                if (stack_pointer < 2) {
                    return 0;
                }
                stack[stack_pointer - 2] = stack[stack_pointer - 2] + stack[stack_pointer - 1];
                --stack_pointer;
                break;
            case RPN_OP_SUB:
                if (stack_pointer < 2) {
                    return 0;
                }
                stack[stack_pointer - 2] = stack[stack_pointer - 2] - stack[stack_pointer - 1];
                --stack_pointer;
                break;
            case RPN_OP_MUL:
                if (stack_pointer < 2) {
                    return 0;
                }
                stack[stack_pointer - 2] = stack[stack_pointer - 2] * stack[stack_pointer - 1];
                --stack_pointer;
                break;
            case RPN_OP_DIV:
                if (stack_pointer < 2) {
                    return 0;
                }
                stack[stack_pointer - 2] = fabsf(stack[stack_pointer - 1]) > 1.0e-8f
                    ? stack[stack_pointer - 2] / stack[stack_pointer - 1]
                    : 0.0f;
                --stack_pointer;
                break;
            case RPN_OP_POW:
                if (stack_pointer < 2) {
                    return 0;
                }
                stack[stack_pointer - 2] = powf(stack[stack_pointer - 2], stack[stack_pointer - 1]);
                --stack_pointer;
                break;
            case RPN_OP_STORE_RESULT:
                if (stack_pointer <= 0) {
                    return 0;
                }
                result = stack[stack_pointer - 1];
                break;
            case RPN_OP_RET:
                *result_out = static_cast<int>(result);
                return 1;
            default:
                return 0;
        }
    }

    *result_out = static_cast<int>(result);
    return stack_pointer > 0 ? 1 : 0;
}
