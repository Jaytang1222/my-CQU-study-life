`timescale 1ns / 1ps

module andgate_sim2();
//input
reg a=0;
reg b=0;
//output
wire c;
andgate #(1) u(a,b,c);
initial begin
#100 a=1;
#100 begin a=0;b=1;end
#100 a=1;
end
endmodule
