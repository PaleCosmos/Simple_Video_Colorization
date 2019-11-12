import React, {Component} from 'react';
import {Box} from "gestalt";
import 'gestalt/dist/gestalt.css';
import "bootstrap/dist/css/bootstrap.css";
import {Button} from 'reactstrap';

import Header from "./components/Header";
import Menu from "./components/Menu";

export default class App extends Component{
  render(){
    return(
      <Box
      column={12}
      color="lightGray"
      display="flex"
      justifyContent="center"
      minHeight={1028}
      >
          <Box maxWidth={960}
          column={12} color ="white" shape ="rounded">
            <Box color="eggplant" shape="roundedTop">
              <Header/>
              <Menu/>
            </Box>
          <Box padding={5} >
            <Button onClick={()=>alert('aaa')}>
              이것은 버튼이다
            </Button>
          </Box>
          </Box>
      </Box>
    )
  }
}