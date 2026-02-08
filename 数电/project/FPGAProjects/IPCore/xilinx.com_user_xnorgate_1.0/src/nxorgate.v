`timescale 1ns / 1ps

module xnorgate
#(parameter WIDTH=8)
(
    input [(WIDTH-1):0] a,
    input [(WIDTH-1):0] b,
    output [(WIDTH-1):0] c
    );
    
    assign c = (a & b) | (~a & ~b);
endmodule