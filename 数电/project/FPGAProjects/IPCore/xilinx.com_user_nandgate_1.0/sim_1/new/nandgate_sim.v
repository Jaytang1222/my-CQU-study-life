`timescale 1ns / 1ps

module nandgate_sim();
    // 32位输入信号（reg类型，用于在initial块中赋值）
    reg [31:0] a;
    reg [31:0] b;
    
    // 32位输出信号（wire类型，由与非门模块驱动）
    wire [31:0] c;
    
    // 例化32位与非门模块（指定WIDTH=32）
    nandgate #(32) uut (
        .a(a),
        .b(b),
        .c(c)
    );
    
    // 仿真激励：覆盖不同输入组合，验证与非逻辑
    initial begin
        // 初始状态：全0输入
        a = 32'h00000000;
        b = 32'h00000000;
        #100;  // 等待100ns
        
        // 情况1：a全1，b全0 → 与非结果应为全1
        a = 32'hFFFFFFFF;
        b = 32'h00000000;
        #100;
        
        // 情况2：a全0，b全1 → 与非结果应为全1
        a = 32'h00000000;
        b = 32'hFFFFFFFF;
        #100;
        
        // 情况3：a全1，b全1 → 与非结果应为全0
        a = 32'hFFFFFFFF;
        b = 32'hFFFFFFFF;
        #100;
        
        // 情况4：部分位为1的组合输入
        a = 32'hA5A5A5A5;  // 二进制：10100101...
        b = 32'h5A5A5A5A;  // 二进制：01011010...
        #100;
        
        // 情况5：另一种部分位组合
        a = 32'h0F0F0F0F;
        b = 32'hF0F0F0F0;
        #100;
        
        $finish;  // 结束仿真
    end
    
endmodule
    