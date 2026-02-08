`timescale 1ns / 1ps

module notgate_sim2();
reg a = 0;

wire c;

notgate #(1) u(a, c);
initial begin
#100 a = 1;
#100 a = 0;
#100 a = 1;
end
endmodule