`timescale 1ns / 1ps
module orgate_sim(
);
    reg [31:0] a = 32'h00000000;
    reg [31:0] b = 32'h00000000;

    wire [31:0] c;

    orgate #(32) u(a, b, c);

    initial begin
        #100 a = 32'hffffffff;
        #100 begin a = 32'h00000000; b = 32'hffffffff; end
        #100 a = 32'h007fa509;
        #100 a = 32'hffffffff;
    end
endmodule