`timescale 1ns / 1ps
module notgate
#(parameter WIDTH=8)
(
    input [(WIDTH-1):0] a,
    output [(WIDTH-1):0] c
);

    assign c = ~a;

endmodule