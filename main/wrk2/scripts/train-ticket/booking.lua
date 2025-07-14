wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- === 사용자 정보 (테스트용) ===
local username = "fdse_microservice"
local password = "111111"

-- === 시나리오 전역 상태 ===
local token = nil
local userId = nil
local trip = {
  trip_id = "G1234",
  from = "Shang Hai",
  to = "Su Zhou",
  seat_type = "2",
  seat_price = "100"
}
local departure_date = "2025-08-01"
local contact_id = nil
local next_step = "login"

-- === 시나리오 동작 ===
request = function()
  if next_step == "login" then
    return wrk.format("POST", "/api/v1/users/login", nil, string.format(
      '{"username":"%s", "password":"%s"}', username, password))

  elseif next_step == "search_ticket" then
    local body = string.format(
      '{"startingPlace":"%s", "endPlace":"%s", "departureTime":"%s"}',
      trip.from, trip.to, departure_date)
    return wrk.format("POST", "/api/v1/travelservice/trips/left", {
      ["Authorization"] = token,
      ["Content-Type"] = "application/json"
    }, body)

  elseif next_step == "start_booking" then
    local url = string.format("/client_ticket_book.html?tripId=%s&from=%s&to=%s&seatType=%s&seat_price=%s&date=%s",
      trip.trip_id, trip.from, trip.to, trip.seat_type, trip.seat_price, departure_date)
    return wrk.format("GET", url, { ["Authorization"] = token })

  elseif next_step == "get_assurance_types" then
    return wrk.format("GET", "/api/v1/assuranceservice/assurances/types", {
      ["Authorization"] = token
    })

  elseif next_step == "get_foods" then
    local url = string.format("/api/v1/foodservice/foods/%s/%s/%s/%s",
      departure_date, trip.from, trip.to, trip.trip_id)
    return wrk.format("GET", url, { ["Authorization"] = token })

  elseif next_step == "select_contact" then
    local url = string.format("/api/v1/contactservice/contacts/account/%s", userId)
    return wrk.format("GET", url, { ["Authorization"] = token })

  elseif next_step == "finish_booking" then
    local body = string.format([[
    {
      "accountId": "%s",
      "contactsId": "%s",
      "tripId": "%s",
      "seatType": "%s",
      "date": "%s",
      "from": "%s",
      "to": "%s",
      "assurance": "0",
      "foodType": 1,
      "foodName": "Bone Soup",
      "foodPrice": 2.5,
      "stationName": "",
      "storeName": ""
    }
    ]], userId, contact_id or userId, trip.trip_id, trip.seat_type,
         departure_date, trip.from, trip.to)

    return wrk.format("POST", "/api/v1/preserveservice/preserve", {
      ["Authorization"] = token,
      ["Content-Type"] = "application/json"
    }, body)

  else
    return wrk.format("GET", "/index.html")
  end
end

-- === 응답 처리 ===
response = function(status, headers, body)
  if next_step == "login" and status == 200 then
    local m = string.match(body, '"token"%s*:%s*"([^"]+)"')
    local uid = string.match(body, '"userId"%s*:%s*"([^"]+)"')
    if m and uid then
      token = "Bearer " .. m
      userId = uid
      next_step = "search_ticket"
    end

  elseif next_step == "search_ticket" then
    next_step = "start_booking"

  elseif next_step == "start_booking" then
    next_step = "get_assurance_types"

  elseif next_step == "get_assurance_types" then
    next_step = "get_foods"

  elseif next_step == "get_foods" then
    next_step = "select_contact"

  elseif next_step == "select_contact" and status == 200 then
    contact_id = string.match(body, '"id"%s*:%s*"([^"]+)"')
    next_step = "finish_booking"

  elseif next_step == "finish_booking" then
    next_step = "done"
  end
end
