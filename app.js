class AvatarWarper {
    constructor() {
        // Initialize PixiJS app
        this.app = new PIXI.Application({
            width: 400,
            height: 400,
            backgroundColor: 0xf0f0f0,
            resolution: window.devicePixelRatio || 1,
            antialias: true,
            autoDensity: true,
        });

        // Wait for DOM to be ready before appending
        if (document.getElementById('app')) {
            document.getElementById('app').appendChild(this.app.view);
        } else {
            console.error('App container not found');
            return;
        }

        // Setup UI elements
        this.cursor = document.getElementById('cursor');
        this.uploadBtn = document.getElementById('upload-btn');
        this.fileInput = document.getElementById('file-input');
        
        // Initialize properties
        this.plane = null;
        this.landmarks = null;
        this.originalLandmarks = null;
        this.centerX = this.app.screen.width / 2;
        this.centerY = this.app.screen.height / 2;
        
        // Face++ API configuration
        this.apiKey = '7x4_pDYsYbBfVKB1f2GQ-_Rlrbas7lQ4';
        this.apiSecret = 'd1R2GFvSAHfhK-XXDEQ0j63mrs4EBIWm';
        this.apiUrl = 'https://api-us.faceplusplus.com/facepp/v3/detect';
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Mouse movement
        document.addEventListener('mousemove', (e) => {
            this.handleMouseMove(e);
        });
        
        // Upload button
        this.uploadBtn.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // File input
        this.fileInput.addEventListener('change', (e) => {
            this.handleImageUpload(e);
        });
    }
    
    handleMouseMove(e) {
        // Update cursor
        this.cursor.style.left = e.clientX + 'px';
        this.cursor.style.top = e.clientY + 'px';
        
        if (this.plane && this.landmarks) {
            const rect = this.app.view.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) / rect.width * this.app.screen.width;
            const mouseY = (e.clientY - rect.top) / rect.height * this.app.screen.height;
            
            this.warpMesh(mouseX, mouseY);
        }
    }
    
    async handleImageUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Load image as base64
        const reader = new FileReader();
        reader.onload = async (event) => {
            const base64 = event.target.result;
            
            // Create texture from base64
            const texture = await PIXI.Texture.fromURL(base64);
            
            // Create or update plane
            if (this.plane) {
                this.app.stage.removeChild(this.plane);
            }
            
            // Create simple plane with 10x10 segments
            this.plane = new PIXI.SimplePlane(texture, 10, 10);
            this.plane.width = this.app.screen.width;
            this.plane.height = this.app.screen.height;
            this.app.stage.addChild(this.plane);
            
            // Detect face landmarks
            await this.detectLandmarks(base64);
        };
        reader.readAsDataURL(file);
    }
    
    async detectLandmarks(base64) {
        try {
            const formData = new FormData();
            formData.append('api_key', this.apiKey);
            formData.append('api_secret', this.apiSecret);
            formData.append('image_base64', base64.split(',')[1]);
            formData.append('return_landmark', '1');
            
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.faces && data.faces.length > 0) {
                this.landmarks = data.faces[0].landmark;
                this.originalLandmarks = JSON.parse(JSON.stringify(this.landmarks));
                
                // Store facial feature regions
                this.setupFacialFeatures();
            } else {
                console.error('No face detected');
            }
        } catch (error) {
            console.error('API Error:', error);
        }
    }
    
    setupFacialFeatures() {
        // Define facial regions
        this.facialFeatures = {
            leftEye: {
                points: this.getLeftEye(),
                center: this.getFeatureCenter(this.getLeftEye()),
                radius: this.getFeatureRadius(this.getLeftEye())
            },
            rightEye: {
                points: this.getRightEye(),
                center: this.getFeatureCenter(this.getRightEye()),
                radius: this.getFeatureRadius(this.getRightEye())
            },
            leftEyebrow: {
                points: this.getLeftEyebrow(),
                center: this.getFeatureCenter(this.getLeftEyebrow()),
                radius: this.getFeatureRadius(this.getLeftEyebrow())
            },
            rightEyebrow: {
                points: this.getRightEyebrow(),
                center: this.getFeatureCenter(this.getRightEyebrow()),
                radius: this.getFeatureRadius(this.getRightEyebrow())
            }
        };
    }
    
    getLeftEye() {
        return [
            'left_eye_left_corner', 'left_eye_upper_left_quarter',
            'left_eye_top', 'left_eye_upper_right_quarter',
            'left_eye_right_corner', 'left_eye_lower_right_quarter',
            'left_eye_bottom', 'left_eye_lower_left_quarter'
        ];
    }
    
    getRightEye() {
        return [
            'right_eye_left_corner', 'right_eye_upper_left_quarter',
            'right_eye_top', 'right_eye_upper_right_quarter',
            'right_eye_right_corner', 'right_eye_lower_right_quarter',
            'right_eye_bottom', 'right_eye_lower_left_quarter'
        ];
    }
    
    getLeftEyebrow() {
        return [
            'left_eyebrow_left_corner',
            'left_eyebrow_upper_left_quarter',
            'left_eyebrow_upper_middle',
            'left_eyebrow_upper_right_quarter',
            'left_eyebrow_right_corner'
        ];
    }
    
    getRightEyebrow() {
        return [
            'right_eyebrow_left_corner',
            'right_eyebrow_upper_left_quarter',
            'right_eyebrow_upper_middle',
            'right_eyebrow_upper_right_quarter',
            'right_eyebrow_right_corner'
        ];
    }
    
    getFeatureCenter(points) {
        const landmarks = points.map(key => this.landmarks[key]);
        const x = landmarks.reduce((sum, p) => sum + p.x, 0) / landmarks.length;
        const y = landmarks.reduce((sum, p) => sum + p.y, 0) / landmarks.length;
        return { x, y };
    }
    
    getFeatureRadius(points) {
        const center = this.getFeatureCenter(points);
        const landmarks = points.map(key => this.landmarks[key]);
        
        // Calculate maximum distance from center to any point
        return Math.max(...landmarks.map(p => 
            Math.sqrt(Math.pow(p.x - center.x, 2) + Math.pow(p.y - center.y, 2))
        ));
    }
    
    warpMesh(mouseX, mouseY) {
        if (!this.plane || !this.landmarks || !this.facialFeatures) return;
        
        const vertices = this.plane.geometry.getBuffer('aVertexPosition').data;
        const totalPoints = vertices.length / 2;
        
        // Calculate global movement vector
        const dx = mouseX - this.centerX;
        const dy = mouseY - this.centerY;
        const angle = Math.atan2(dy, dx);
        const distance = Math.min(Math.sqrt(dx * dx + dy * dy) / 200, 1);
        
        // Reset vertices to their original positions
        for (let i = 0; i < totalPoints; i++) {
            const vertexX = (i % 11) / 10;
            const vertexY = Math.floor(i / 11) / 10;
            
            // Initialize warp values
            let totalWarpX = 0;
            let totalWarpY = 0;
            let totalInfluence = 0;
            
            // Only process eyes (not eyebrows) for more natural movement
            ['leftEye', 'rightEye'].forEach(featureKey => {
                const feature = this.facialFeatures[featureKey];
                const normalizedCenter = {
                    x: feature.center.x / this.app.screen.width,
                    y: feature.center.y / this.app.screen.height
                };
                
                // Calculate distance from vertex to eye center
                const dx = vertexX - normalizedCenter.x;
                const dy = vertexY - normalizedCenter.y;
                const distToEye = Math.sqrt(dx * dx + dy * dy);
                
                // Adjust radius for tighter control
                const normalizedRadius = (feature.radius / this.app.screen.width) * 1.2;
                
                if (distToEye <= normalizedRadius) {
                    // Calculate influence with smooth falloff
                    const influence = Math.pow(1 - (distToEye / normalizedRadius), 2);
                    
                    // Calculate direction to cursor
                    const toCursorX = mouseX - (normalizedCenter.x * this.app.screen.width);
                    const toCursorY = mouseY - (normalizedCenter.y * this.app.screen.height);
                    const cursorDist = Math.sqrt(toCursorX * toCursorX + toCursorY * toCursorY);
                    
                    // Normalize and scale the movement
                    const maxMove = 10; // Maximum pixels to move
                    const moveScale = Math.min(cursorDist / 200, 1) * maxMove;
                    
                    // Add to total warp
                    totalWarpX += (toCursorX / cursorDist) * moveScale * influence;
                    totalWarpY += (toCursorY / cursorDist) * moveScale * influence;
                    totalInfluence += influence;
                }
            });
            
            // Apply warping if there's any influence
            if (totalInfluence > 0) {
                vertices[i * 2] = vertexX * this.app.screen.width + (totalWarpX / totalInfluence);
                vertices[i * 2 + 1] = vertexY * this.app.screen.height + (totalWarpY / totalInfluence);
            } else {
                vertices[i * 2] = vertexX * this.app.screen.width;
                vertices[i * 2 + 1] = vertexY * this.app.screen.height;
            }
        }
        
        // Update the mesh
        this.plane.geometry.getBuffer('aVertexPosition').update();
    }
    
    getFeatureInfluence(x, y, center, radius) {
        const dx = x - center.x;
        const dy = y - center.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Increase the influence radius slightly
        const influenceRadius = radius * 1.5;
        
        // Smooth falloff
        if (distance > influenceRadius) return 0;
        return Math.pow(1 - (distance / influenceRadius), 2);
    }
}

// Initialize the app when both DOM and PIXI are ready
function init() {
    if (typeof PIXI === 'undefined') {
        console.error('PIXI.js not loaded');
        return;
    }

    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => new AvatarWarper());
    } else {
        new AvatarWarper();
    }
}

// Check if PIXI is loaded every 100ms for up to 5 seconds
let attempts = 0;
const maxAttempts = 50;
const checkPixi = setInterval(() => {
    if (typeof PIXI !== 'undefined') {
        clearInterval(checkPixi);
        init();
    } else if (++attempts >= maxAttempts) {
        clearInterval(checkPixi);
        console.error('Failed to load PIXI.js');
    }
}, 100); 