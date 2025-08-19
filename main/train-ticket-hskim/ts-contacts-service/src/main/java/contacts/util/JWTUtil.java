package contacts.util;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import javax.crypto.SecretKey;

/**
 * JWT 토큰 검증을 위한 유틸리티 클래스
 * ts-sso-service와 동일한 시크릿 키를 사용
 */
public class JWTUtil {
    
    // ts-sso-service와 동일한 시크릿 키
    private static final String JWT_SECRET_KEY = "train-ticket-microservices-jwt-secret-key-2024";
    private static final SecretKey FIXED_JWT_KEY = Keys.hmacShaKeyFor(JWT_SECRET_KEY.getBytes());
    
    /**
     * JWT 토큰 검증
     */
    public static boolean verifyJWTToken(String token) {
        try {
            Jwts.parserBuilder()
                .setSigningKey(FIXED_JWT_KEY)
                .build()
                .parseClaimsJws(token);
            return true;
        } catch (Exception e) {
            System.out.println("[Contacts Service][JWT] Token verification failed: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * JWT 토큰에서 사용자 ID 추출
     */
    public static String extractUserIdFromJWT(String token) {
        try {
            Claims claims = Jwts.parserBuilder()
                .setSigningKey(FIXED_JWT_KEY)
                .build()
                .parseClaimsJws(token)
                .getBody();
            return claims.getSubject();
        } catch (Exception e) {
            System.out.println("[Contacts Service][JWT] Failed to extract user ID: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * JWT 토큰에서 이메일 추출
     */
    public static String extractEmailFromJWT(String token) {
        try {
            Claims claims = Jwts.parserBuilder()
                .setSigningKey(FIXED_JWT_KEY)
                .build()
                .parseClaimsJws(token)
                .getBody();
            return claims.get("email", String.class);
        } catch (Exception e) {
            System.out.println("[Contacts Service][JWT] Failed to extract email: " + e.getMessage());
            return null;
        }
    }
}
