package sso.service;

import org.springframework.http.HttpHeaders;
import sso.domain.*;

public interface AccountSsoService {

    RegisterResult create(RegisterInfo ri, HttpHeaders headers);

    Account createAccount(Account account, HttpHeaders headers);

    LoginResult login(LoginInfo li, HttpHeaders headers);

    PutLoginResult loginPutToken(String loginToken);

    LogoutResult logoutDeleteToken(LogoutInfo li, HttpHeaders headers);

    VerifyResult verifyLoginToken(String verifyToken, HttpHeaders headers);

    FindAllAccountResult findAllAccount(HttpHeaders headers);

    GetLoginAccountList findAllLoginAccount(HttpHeaders headers);

    ModifyAccountResult saveChanges(ModifyAccountInfo modifyAccountInfo, HttpHeaders headers);

    GetAccountByIdResult getAccountById(GetAccountByIdInfo info, HttpHeaders headers);

    Contacts adminLogin(String name, String password, HttpHeaders headers);

    DeleteAccountResult deleteAccount(String accountId, HttpHeaders headers);

    // JWT 토큰 검증 메서드
    boolean verifyJWTToken(String token);

    // JWT 토큰에서 사용자 ID 추출 메서드
    String extractUserIdFromJWT(String token);

}
