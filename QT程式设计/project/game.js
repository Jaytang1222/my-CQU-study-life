// 游戏常量
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const TILE_SIZE = 40;
const PLAYER_SPEED = 3;
const BOMB_RANGE = 3;
const BOMB_TIMER = 3000;
const EXPLOSION_DURATION = 500;

// 游戏对象
let canvas, ctx;
let player1; // 人类玩家
let player2; // AI玩家
let bombs = [];
let explosions = [];
let enemies = [];
let blocks = [];
let map;
let score1 = 0; // 人类玩家分数
let score2 = 0; // AI玩家分数
let lives1 = 3; // 人类玩家生命
let lives2 = 3; // AI玩家生命

// 方向常量
const DIRECTION = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3
};

// 初始化游戏
function init() {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');
    
    // 创建地图
    createMap();
    
    // 创建玩家
    player1 = new Player(1 * TILE_SIZE, 1 * TILE_SIZE, 'player1'); // 人类玩家
    player2 = new AIPlayer(18 * TILE_SIZE, 12 * TILE_SIZE, 'player2'); // AI玩家
    
    // 设置键盘控制
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    
    // 开始游戏循环
    gameLoop();
}

// 玩家类
class Player {
    constructor(x, y, id) {
        this.x = x;
        this.y = y;
        this.width = TILE_SIZE - 4;
        this.height = TILE_SIZE - 4;
        this.speed = PLAYER_SPEED;
        this.direction = DIRECTION.DOWN;
        this.keys = {};
        this.canPlaceBomb = true;
        this.id = id; // 玩家ID，用于区分玩家1和玩家2
    }
    
    update() {
        // 移动处理
        if (this.keys['ArrowUp'] || this.keys['w'] || this.keys['W']) {
            this.move(0, -this.speed);
            this.direction = DIRECTION.UP;
        }
        if (this.keys['ArrowRight'] || this.keys['d'] || this.keys['D']) {
            this.move(this.speed, 0);
            this.direction = DIRECTION.RIGHT;
        }
        if (this.keys['ArrowDown'] || this.keys['s'] || this.keys['S']) {
            this.move(0, this.speed);
            this.direction = DIRECTION.DOWN;
        }
        if (this.keys['ArrowLeft'] || this.keys['a'] || this.keys['A']) {
            this.move(-this.speed, 0);
            this.direction = DIRECTION.LEFT;
        }
        
        // 炸弹放置冷却
        if (!this.canPlaceBomb) {
            setTimeout(() => {
                this.canPlaceBomb = true;
            }, 500);
        }
    }
    
    move(dx, dy) {
        // 检查移动是否合法
        if (!this.checkCollision(this.x + dx, this.y, dx, dy)) {
            this.x += dx;
        }
        if (!this.checkCollision(this.x, this.y + dy, dx, dy)) {
            this.y += dy;
        }
    }
    
    checkCollision(x, y, dx, dy) {
        // 边界检查
        if (x < 0 || x + this.width > CANVAS_WIDTH || y < 0 || y + this.height > CANVAS_HEIGHT) {
            return true;
        }
        
        // 砖块碰撞
        for (let block of blocks) {
            if (block.solid && 
                x < block.x + block.width &&
                x + this.width > block.x &&
                y < block.y + block.height &&
                y + this.height > block.y) {
                return true;
            }
        }
        
        return false;
    }
    
    placeBomb() {
        if (this.canPlaceBomb) {
            // 将炸弹放置在网格中心
            let bombX = Math.floor(this.x / TILE_SIZE) * TILE_SIZE;
            let bombY = Math.floor(this.y / TILE_SIZE) * TILE_SIZE;
            
            bombs.push(new Bomb(bombX, bombY));
            this.canPlaceBomb = false;
        }
    }
    
    draw() {
        // 根据玩家ID设置不同颜色
        if (this.id === 'player1') {
            ctx.fillStyle = '#00ff00';
        } else {
            ctx.fillStyle = '#0000ff';
        }
        ctx.fillRect(this.x, this.y, this.width, this.height);
        
        // 绘制方向指示器
        if (this.id === 'player1') {
            ctx.fillStyle = '#008800';
        } else {
            ctx.fillStyle = '#000088';
        }
        let centerX = this.x + this.width / 2;
        let centerY = this.y + this.height / 2;
        
        switch(this.direction) {
            case DIRECTION.UP:
                ctx.fillRect(centerX - 4, centerY - 12, 8, 8);
                break;
            case DIRECTION.RIGHT:
                ctx.fillRect(centerX + 4, centerY - 4, 8, 8);
                break;
            case DIRECTION.DOWN:
                ctx.fillRect(centerX - 4, centerY + 4, 8, 8);
                break;
            case DIRECTION.LEFT:
                ctx.fillRect(centerX - 12, centerY - 4, 8, 8);
                break;
        }
    }
}

// 炸弹类
class Bomb {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.width = TILE_SIZE;
        this.height = TILE_SIZE;
        this.timer = BOMB_TIMER;
        this.placedAt = Date.now();
        this.range = BOMB_RANGE;
    }
    
    update() {
        let now = Date.now();
        if (now - this.placedAt > this.timer) {
            this.explode();
        }
    }
    
    explode() {
        // 创建爆炸效果
        explosions.push(new Explosion(this.x, this.y, this.range));
        
        // 移除炸弹
        bombs.splice(bombs.indexOf(this), 1);
    }
    
    draw() {
        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.arc(this.x + TILE_SIZE/2, this.y + TILE_SIZE/2, TILE_SIZE/3, 0, Math.PI * 2);
        ctx.fill();
        
        // 绘制炸弹倒计时
        let progress = (Date.now() - this.placedAt) / this.timer;
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(this.x + TILE_SIZE/2, this.y + TILE_SIZE/2, TILE_SIZE/2 - 2, 0, Math.PI * 2 * progress);
        ctx.stroke();
    }
}

// 爆炸类
class Explosion {
    constructor(x, y, range) {
        this.x = x;
        this.y = y;
        this.range = range;
        this.startTime = Date.now();
        this.duration = EXPLOSION_DURATION;
        this.particles = this.createParticles();
        
        // 检查爆炸范围内的对象
        this.checkExplosionEffects();
    }
    
    createParticles() {
        let particles = [];
        // 中心粒子
        particles.push({x: this.x + TILE_SIZE/2, y: this.y + TILE_SIZE/2});
        
        // 四个方向的粒子
        for (let dir = 0; dir < 4; dir++) {
            for (let i = 1; i <= this.range; i++) {
                let px = this.x + TILE_SIZE/2;
                let py = this.y + TILE_SIZE/2;
                
                switch(dir) {
                    case DIRECTION.UP:
                        py -= TILE_SIZE * i;
                        break;
                    case DIRECTION.RIGHT:
                        px += TILE_SIZE * i;
                        break;
                    case DIRECTION.DOWN:
                        py += TILE_SIZE * i;
                        break;
                    case DIRECTION.LEFT:
                        px -= TILE_SIZE * i;
                        break;
                }
                
                // 检查是否碰到实心砖块
                if (this.checkBlockCollision(px, py)) {
                    break;
                }
                
                particles.push({x: px, y: py});
            }
        }
        
        return particles;
    }
    
    checkBlockCollision(x, y) {
        let gridX = Math.floor(x / TILE_SIZE);
        let gridY = Math.floor(y / TILE_SIZE);
        
        for (let block of blocks) {
            if (block.solid && 
                gridX * TILE_SIZE === block.x && 
                gridY * TILE_SIZE === block.y) {
                // 摧毁可破坏的砖块
                if (block.destructible) {
                    blocks.splice(blocks.indexOf(block), 1);
                    score += 10;
                    updateScore();
                }
                return true;
            }
        }
        
        return false;
    }
    
    checkExplosionEffects() {
        // 检查玩家是否在爆炸范围内
        for (let particle of this.particles) {
            // 检查玩家1
            if (player1.x < particle.x + TILE_SIZE/2 &&
                player1.x + player1.width > particle.x - TILE_SIZE/2 &&
                player1.y < particle.y + TILE_SIZE/2 &&
                player1.y + player1.height > particle.y - TILE_SIZE/2) {
                playerHit(player1);
            }
            
            // 检查玩家2
            if (player2.x < particle.x + TILE_SIZE/2 &&
                player2.x + player2.width > particle.x - TILE_SIZE/2 &&
                player2.y < particle.y + TILE_SIZE/2 &&
                player2.y + player2.height > particle.y - TILE_SIZE/2) {
                playerHit(player2);
            }
        }
    }
    
    update() {
        let now = Date.now();
        if (now - this.startTime > this.duration) {
            explosions.splice(explosions.indexOf(this), 1);
        }
    }
    
    draw() {
        ctx.fillStyle = '#ff8800';
        
        for (let particle of this.particles) {
            ctx.fillRect(particle.x - TILE_SIZE/2, particle.y - TILE_SIZE/2, TILE_SIZE, TILE_SIZE);
        }
    }
}

// 敌人类
class Enemy {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.width = TILE_SIZE - 4;
        this.height = TILE_SIZE - 4;
        this.speed = 1;
        this.direction = Math.floor(Math.random() * 4);
        this.moveTimer = 0;
        this.moveInterval = 200;
    }
    
    // 在Enemy类的update方法中添加追踪玩家的逻辑
update() {
    this.moveTimer++;
    
    if (this.moveTimer > this.moveInterval) {
        this.moveTimer = 0;
        
        // 增强：追踪玩家的简单逻辑
        let dx = player.x - this.x;
        let dy = player.y - this.y;
        
        // 计算与玩家的距离
        let distance = Math.sqrt(dx * dx + dy * dy);
        
        // 如果玩家在一定范围内，尝试朝玩家方向移动
        if (distance < TILE_SIZE * 8) {
            if (Math.abs(dx) > Math.abs(dy)) {
                this.direction = dx > 0 ? DIRECTION.RIGHT : DIRECTION.LEFT;
            } else {
                this.direction = dy > 0 ? DIRECTION.DOWN : DIRECTION.UP;
            }
        } else if (Math.random() < 0.3) {
            // 随机改变方向
            this.direction = Math.floor(Math.random() * 4);
        }
        
        let moveX = 0, moveY = 0;
        switch(this.direction) {
            case DIRECTION.UP:
                moveY = -this.speed;
                break;
            case DIRECTION.RIGHT:
                moveX = this.speed;
                break;
            case DIRECTION.DOWN:
                moveY = this.speed;
                break;
            case DIRECTION.LEFT:
                moveX = -this.speed;
                break;
        }
        
        // 尝试移动，如果失败则改变方向
        if (!this.checkCollision(this.x + moveX, this.y + moveY)) {
            this.x += moveX;
            this.y += moveY;
        } else {
            this.direction = (this.direction + 1) % 4;
        }
    }
    
    // 检查是否碰到玩家
    if (this.checkPlayerCollision()) {
        playerHit();
    }
}
    
    checkCollision(x, y) {
        // 边界检查
        if (x < 0 || x + this.width > CANVAS_WIDTH || y < 0 || y + this.height > CANVAS_HEIGHT) {
            return true;
        }
        
        // 砖块碰撞
        for (let block of blocks) {
            if (block.solid && 
                x < block.x + block.width &&
                x + this.width > block.x &&
                y < block.y + block.height &&
                y + this.height > block.y) {
                return true;
            }
        }
        
        return false;
    }
    
    checkPlayerCollision() {
        return player.x < this.x + this.width &&
               player.x + player.width > this.x &&
               player.y < this.y + this.height &&
               player.y + player.height > this.y;
    }
    
    draw() {
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(this.x, this.y, this.width, this.height);
    }
}

// 砖块类
class Block {
    constructor(x, y, solid = true, destructible = false) {
        this.x = x;
        this.y = y;
        this.width = TILE_SIZE;
        this.height = TILE_SIZE;
        this.solid = solid;
        this.destructible = destructible;
    }
    
    draw() {
        if (this.solid) {
            if (this.destructible) {
                ctx.fillStyle = '#884400';
            } else {
                ctx.fillStyle = '#333333';
            }
            ctx.fillRect(this.x, this.y, this.width, this.height);
        }
    }
}

// 创建地图
function createMap() {
    // 地图数据: 0=空, 1=实心墙, 2=可破坏墙
    let mapData = [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,2,1,2,1,2,0,1,0,2,1,2,1,2,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,2,1,2,1,2,0,1,0,2,1,2,1,2,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,2,1,2,1,2,0,1,0,2,1,2,1,2,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ];
    
    for (let row = 0; row < mapData.length; row++) {
        for (let col = 0; col < mapData[row].length; col++) {
            if (mapData[row][col] === 1) {
                blocks.push(new Block(col * TILE_SIZE, row * TILE_SIZE, true, false));
            } else if (mapData[row][col] === 2) {
                blocks.push(new Block(col * TILE_SIZE, row * TILE_SIZE, true, true));
            }
        }
    }
}

// 创建敌人 (在对战模式中不使用)
function createEnemies() {
    // 对战模式下不需要敌人
}

// 处理键盘按下
function handleKeyDown(e) {
    player1.keys[e.key] = true;
    
    if (e.key === ' ') {
        player1.placeBomb();
    }
}

// 处理键盘松开
function handleKeyUp(e) {
    player1.keys[e.key] = false;
}

// 玩家被击中
function playerHit(player) {
    if (player.id === 'player1') {
        lives1--;
        updateLives();
        
        // 重置玩家1位置
        player1.x = 1 * TILE_SIZE;
        player1.y = 1 * TILE_SIZE;
        
        if (lives1 <= 0) {
            gameOver('player2'); // AI玩家获胜
        }
    } else {
        lives2--;
        updateLives();
        
        // 重置玩家2位置
        player2.x = 18 * TILE_SIZE;
        player2.y = 12 * TILE_SIZE;
        
        if (lives2 <= 0) {
            gameOver('player1'); // 人类玩家获胜
        }
    }
}

// 敌人被击中 (暂时保留但不使用)
function enemyHit(enemy) {
    enemies.splice(enemies.indexOf(enemy), 1);
    
    // 为击中敌人的玩家增加分数
    // 这里需要额外的逻辑来判断是哪个玩家的炸弹击中了敌人
    // 简化处理，暂时不使用
}

// AI玩家类
class AIPlayer extends Player {
    constructor(x, y, id) {
        super(x, y, id);
        this.speed = PLAYER_SPEED;
        this.decisionTimer = 0;
        this.decisionInterval = 50; // AI决策间隔
        this.target = null;
    }
    
    update() {
        this.decisionTimer++;
        
        if (this.decisionTimer > this.decisionInterval) {
            this.decisionTimer = 0;
            this.makeDecision();
        }
        
        // 炸弹放置冷却
        if (!this.canPlaceBomb) {
            setTimeout(() => {
                this.canPlaceBomb = true;
            }, 500);
        }
        
        // 根据决策移动
        this.moveAccordingToDecision();
    }
    
    makeDecision() {
        // AI决策逻辑
        
        // 1. 设置目标为玩家1
        this.target = player1;
        
        // 2. 检查是否需要放置炸弹
        this.checkPlaceBomb();
        
        // 3. 确定移动方向
        this.determineMoveDirection();
    }
    
    checkPlaceBomb() {
        // 计算与目标的距离
        let dx = this.target.x - this.x;
        let dy = this.target.y - this.y;
        let distance = Math.sqrt(dx * dx + dy * dy);
        
        // 如果目标在爆炸范围内，放置炸弹
        if (distance < TILE_SIZE * BOMB_RANGE && this.canPlaceBomb) {
            this.placeBomb();
        }
        
        // 检查是否在一条直线上且可以攻击到目标
        if (Math.abs(dx) < TILE_SIZE && Math.abs(dy) < TILE_SIZE * BOMB_RANGE) {
            // 垂直方向
            let canHit = true;
            let startY = Math.min(this.y, this.target.y) + TILE_SIZE;
            let endY = Math.max(this.y, this.target.y) - TILE_SIZE;
            
            for (let y = startY; y <= endY; y += TILE_SIZE) {
                let gridX = Math.floor(this.x / TILE_SIZE);
                let gridY = Math.floor(y / TILE_SIZE);
                
                for (let block of blocks) {
                    if (block.solid && 
                        gridX * TILE_SIZE === block.x && 
                        gridY * TILE_SIZE === block.y) {
                        canHit = false;
                        break;
                    }
                }
                if (!canHit) break;
            }
            
            if (canHit && this.canPlaceBomb) {
                this.placeBomb();
            }
        } else if (Math.abs(dy) < TILE_SIZE && Math.abs(dx) < TILE_SIZE * BOMB_RANGE) {
            // 水平方向
            let canHit = true;
            let startX = Math.min(this.x, this.target.x) + TILE_SIZE;
            let endX = Math.max(this.x, this.target.x) - TILE_SIZE;
            
            for (let x = startX; x <= endX; x += TILE_SIZE) {
                let gridX = Math.floor(x / TILE_SIZE);
                let gridY = Math.floor(this.y / TILE_SIZE);
                
                for (let block of blocks) {
                    if (block.solid && 
                        gridX * TILE_SIZE === block.x && 
                        gridY * TILE_SIZE === block.y) {
                        canHit = false;
                        break;
                    }
                }
                if (!canHit) break;
            }
            
            if (canHit && this.canPlaceBomb) {
                this.placeBomb();
            }
        }
    }
    
    determineMoveDirection() {
        // 简单的追踪逻辑
        let dx = this.target.x - this.x;
        let dy = this.target.y - this.y;
        
        // 随机选择一个方向，优先选择更接近目标的方向
        if (Math.abs(dx) > Math.abs(dy) || Math.random() < 0.3) {
            if (dx > 0) {
                this.direction = DIRECTION.RIGHT;
            } else {
                this.direction = DIRECTION.LEFT;
            }
        } else {
            if (dy > 0) {
                this.direction = DIRECTION.DOWN;
            } else {
                this.direction = DIRECTION.UP;
            }
        }
        
        // 检查是否会撞到障碍物
        let moveX = 0, moveY = 0;
        switch(this.direction) {
            case DIRECTION.UP:
                moveY = -this.speed;
                break;
            case DIRECTION.RIGHT:
                moveX = this.speed;
                break;
            case DIRECTION.DOWN:
                moveY = this.speed;
                break;
            case DIRECTION.LEFT:
                moveX = -this.speed;
                break;
        }
        
        if (this.checkCollision(this.x + moveX, this.y + moveY)) {
            // 如果会撞到障碍物，随机改变方向
            this.direction = Math.floor(Math.random() * 4);
        }
    }
    
    moveAccordingToDecision() {
        let moveX = 0, moveY = 0;
        
        switch(this.direction) {
            case DIRECTION.UP:
                moveY = -this.speed;
                break;
            case DIRECTION.RIGHT:
                moveX = this.speed;
                break;
            case DIRECTION.DOWN:
                moveY = this.speed;
                break;
            case DIRECTION.LEFT:
                moveX = -this.speed;
                break;
        }
        
        // 尝试移动
        if (!this.checkCollision(this.x + moveX, this.y)) {
            this.x += moveX;
        }
        if (!this.checkCollision(this.x, this.y + moveY)) {
            this.y += moveY;
        }
    }
}

// 更新分数
function updateScore() {
    document.getElementById('score1').textContent = score1;
    document.getElementById('score2').textContent = score2;
}

// 更新生命
function updateLives() {
    document.getElementById('lives1').textContent = lives1;
    document.getElementById('lives2').textContent = lives2;
}

// 游戏结束
function gameOver(winner) {
    let message;
    if (winner === 'player1') {
        message = '恭喜！你赢了！';
        score1++;
    } else {
        message = 'AI赢了！继续努力！';
        score2++;
    }
    alert(message);
    
    // 重置游戏
    lives1 = 3;
    lives2 = 3;
    updateScore();
    updateLives();
    
    // 清空所有对象
    bombs = [];
    explosions = [];
    enemies = [];
    blocks = [];
    
    // 重新创建游戏
    createMap();
    
    // 重置玩家位置
    player1.x = 1 * TILE_SIZE;
    player1.y = 1 * TILE_SIZE;
    player2.x = 18 * TILE_SIZE;
    player2.y = 12 * TILE_SIZE;
}

// 游戏循环
function gameLoop() {
    // 清空画布
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // 绘制砖块
    for (let block of blocks) {
        block.draw();
    }
    
    // 更新和绘制玩家1
    player1.update();
    player1.draw();
    
    // 更新和绘制玩家2
    player2.update();
    player2.draw();
    
    // 更新和绘制炸弹
    for (let bomb of bombs) {
        bomb.update();
        bomb.draw();
    }
    
    // 更新和绘制爆炸
    for (let explosion of explosions) {
        explosion.update();
        explosion.draw();
    }
    
    // 继续游戏循环
    requestAnimationFrame(gameLoop);
}

// 页面加载完成后初始化游戏
window.addEventListener('load', init);