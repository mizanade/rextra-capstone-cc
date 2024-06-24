# API Reference
This API provides endpoints for user registration, login, post recomendation, get recomendation, get all categories, get all activities, get activity by id, get user profie, update user, update profile image.

## Port or Base URL

xxxxxx.com

## Register an Account
Registers a new user.

### Request

```http
  POST /api/v1/auth/register
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `name` | `string` | "your_name" |
| `email` | `email` | "your_email@example.com" |
| `password` | `string` | "your_password" |
| `confirmPassword` | `string` | "your_password" |

### Response (Success - 201 Created):

````
{
  "message": "User account created successfully."
}
}
````

## Login to an Account
Authenticates a user and returns a JWT.

### Request

```http
  POST /api/v1/auth/login
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `email` | `email` | "your_email@example.com" |
| `password` | `string` | "your_password" |

### Response (Success - 200 OK):

````
{
  "token": "your_jwt_token"
}
````
![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/47b3c490-0693-4801-bfdc-fec1178bc509)

## Post Recomendation

![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/b44e8d76-0b0e-43b2-bea2-3b21daca5c66)

## Get Recomendation Result
![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/32b9a3c9-88d6-4436-b05f-00009f4a0cc9)

## Get All Categories
![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/4ac03045-a3ee-41e2-b66a-18fa0ae59136)

## Get All Activities
![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/ee4c902d-401c-4236-8445-58e764795fc5)

## Get Activity by id
![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/76b2cede-f9b8-4946-b125-cdeec83c766f)

## Get User Profile
![image](https://github.com/mizanade/rextra-capstone-cc/assets/129030131/c4797c4c-2241-4b33-9000-4c9b109232be)


## Logout of an Account
Logs out the user. (Client-side: Please remove the stored JWT)

```http
  POST /api/logout
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `-` | `-` | - |


| `-` | `-` | - |
